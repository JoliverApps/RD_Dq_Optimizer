"""Check local signing configuration and its place in the build sequence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build
import debug_signing as signing


class DebugSigningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"LOCALAPPDATA": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def configure(self, thumbprint="A" * 40):
        path = signing.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"thumbprint": thumbprint}))

    def test_unconfigured_build_does_not_sign_or_change_trust(self):
        with patch.object(signing.subprocess, "run") as run:
            self.assertIsNone(signing.sign_debug_executable(self.root / "app.exe"))
            run.assert_not_called()

    def test_invalid_thumbprint_cannot_reach_a_command(self):
        self.configure("'; Remove-Item anything; '")
        with patch.object(signing.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                signing.sign_debug_executable(self.root / "app.exe")
            run.assert_not_called()

    def test_signing_uses_current_user_store_then_verifies(self):
        self.configure("a" * 40)
        with patch.object(signing, "find_signtool", return_value=Path("signtool.exe")):
            with patch.object(signing.subprocess, "run") as run:
                result = signing.sign_debug_executable(self.root / "app.exe")
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][1], "sign")
        self.assertIn("A" * 40, commands[0])
        self.assertNotIn("/sm", commands[0])
        self.assertEqual(commands[1][1:4], ["verify", "/pa", "/v"])
        self.assertEqual(result["mode"], "local_debug")

    def test_declined_setup_does_not_create_or_trust_a_certificate(self):
        with patch.object(signing, "find_signtool", return_value=Path("signtool.exe")):
            with patch("builtins.input", return_value="no"):
                with patch.object(signing, "powershell") as powershell:
                    signing.setup()
                    powershell.assert_not_called()
        self.assertFalse(signing.settings_path().exists())

    def test_setup_records_only_public_thumbprint_after_confirmation(self):
        with patch.object(signing, "find_signtool", return_value=Path("signtool.exe")):
            with patch("builtins.input", return_value="TRUST"):
                with patch.object(
                    signing, "powershell", return_value=json.dumps({"thumbprint": "B" * 40})
                ) as ps:
                    signing.setup()
        self.assertEqual(
            json.loads(signing.settings_path().read_text()), {"thumbprint": "B" * 40}
        )
        script = ps.call_args.args[0]
        self.assertIn("NonExportable", script)
        self.assertNotIn("LocalMachine", script)
        self.assertNotIn("Export-PfxCertificate", script)

    def check_build(self, configuration):
        executable = self.root / "dist" / configuration
        executable = executable / (
            "RD_EQD2_Optimizer.exe" if configuration == "Release"
            else "RD_EQD2_Optimizer/RD_EQD2_Optimizer.exe"
        )
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"test executable")
        events = []

        def self_test(command, report):
            events.append("exe_test" if command[0] == executable else "source_test")

        def sign(path):
            self.assertEqual(path, executable)
            events.append("sign")
            return {"mode": "local_debug"}
        with (
            patch.object(build, "ROOT", self.root),
            patch.object(build.os, "name", "nt"),
            patch.object(build.sys, "version_info", (3, 12)),
            patch.object(build.sys, "argv", ["build.py", "Build", configuration]),
            patch.object(build, "prepare_environment", return_value="python.exe"),
            patch.object(build, "run"),
            patch.object(build, "self_test", side_effect=self_test),
            patch.object(build, "sign_debug_executable", side_effect=sign) as signer,
            patch.object(build, "file_sha256", return_value="test hash"),
            patch.object(build.subprocess, "check_output", return_value=""),
        ):
            self.assertEqual(build.main(), 0)
        manifest = json.loads(
            (self.root / "dist" / configuration / "build-info.json").read_text()
        )
        return events, signer, manifest

    def test_debug_signs_before_executable_verification(self):
        events, signer, manifest = self.check_build("Debug")
        self.assertEqual(events, ["source_test", "sign", "exe_test"])
        signer.assert_called_once()
        self.assertEqual(manifest["signing"], {"mode": "local_debug"})

    def test_release_does_not_use_the_development_certificate(self):
        events, signer, manifest = self.check_build("Release")
        self.assertEqual(events, ["source_test", "exe_test"])
        signer.assert_not_called()
        self.assertIsNone(manifest["signing"])


if __name__ == "__main__":
    unittest.main()
