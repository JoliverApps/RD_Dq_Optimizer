"""Create a local development certificate and sign Debug executables."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def settings_path():
    return Path(os.environ["LOCALAPPDATA"]) / "RD_EQD2_Optimizer" / "debug-signing.json"


def certificate_thumbprint():
    path = settings_path()
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["thumbprint"]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{40}", value):
            raise ValueError("Invalid certificate thumbprint")
        return value.upper()
    except (KeyError, ValueError, TypeError) as error:
        raise RuntimeError(f"Invalid signing settings in {path}: {error}") from error


def find_signtool():
    found = shutil.which("signtool.exe")
    if found:
        return Path(found)
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    candidates = list((kits / "Windows Kits" / "10" / "bin").glob("*/x64/signtool.exe"))
    if candidates:

        def version(path):
            return tuple(int(part) for part in re.findall(r"\d+", path.parent.parent.name))

        return max(candidates, key=version)
    raise RuntimeError(
        "SignTool was not found. In Visual Studio Installer, modify the installation "
        "and add a Windows 10 or Windows 11 SDK, then retry."
    )


def powershell(script):
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'; " + script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "PowerShell failed.")
    return result.stdout.strip()


def setup():
    """Ask for current-user trust, then create or reuse a non-exportable key."""
    find_signtool()
    thumbprint = certificate_thumbprint() or ""
    print(
        "This creates a certificate for local RD EQD2 Optimizer Debug builds.\n"
        "Its non-exportable private key stays in your Windows user certificate store.\n"
        "The public certificate will be trusted in your CURRENT USER Root and\n"
        "TrustedPublisher stores. Applications signed with this key will be trusted\n"
        "under that account. This does not configure other computers or public releases.\n"
        "Smart App Control may still block a locally signed application."
    )
    if input("Type TRUST to create/trust this development certificate: ").strip() != "TRUST":
        print("Cancelled; certificate stores and signing settings were not changed.")
        return
    script = r"""
$cert = $null
if ('THUMBPRINT') {
    $cert = Get-Item 'Cert:\CurrentUser\My\THUMBPRINT' -ErrorAction SilentlyContinue
}
if (-not $cert -or -not $cert.HasPrivateKey -or $cert.NotAfter -le (Get-Date)) {
    $cert = New-SelfSignedCertificate -Type CodeSigningCert `
        -Subject 'CN=RD EQD2 Optimizer Local Debug' `
        -FriendlyName 'RD EQD2 Optimizer Local Debug' `
        -CertStoreLocation 'Cert:\CurrentUser\My' `
        -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 `
        -KeyExportPolicy NonExportable -NotAfter (Get-Date).AddYears(1)
}
$public = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($cert.RawData)
foreach ($name in @('Root', 'TrustedPublisher')) {
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new($name, 'CurrentUser')
    try {
        $store.Open('ReadWrite')
        $store.Add($public)
    } finally {
        $store.Close()
    }
}
@{thumbprint=$cert.Thumbprint} | ConvertTo-Json -Compress
""".replace("THUMBPRINT", thumbprint)
    result = json.loads(powershell(script))
    value = result.get("thumbprint", "")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{40}", value):
        raise RuntimeError("Windows did not return a valid certificate thumbprint.")
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"thumbprint": value.upper()}, indent=2), encoding="utf-8")
    print("Debug signing configured. Certificate thumbprint: " + value.upper())
    print("Now run: .\\Build.cmd Build Debug")


def sign_debug_executable(executable):
    """Sign with the configured current-user key and verify Authenticode trust."""
    thumbprint = certificate_thumbprint()
    if thumbprint is None:
        print("Debug signing is not configured. Setup: py -3.12 tools\\debug_signing.py setup")
        return None
    signtool = str(find_signtool())
    for command in (
        [
            signtool, "sign", "/fd", "SHA256", "/s", "My", "/sha1", thumbprint,
            "/d", "RD EQD2 Optimizer (Local Debug)", str(executable),
        ],
        [signtool, "verify", "/pa", "/v", str(executable)],
    ):
        print("> " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
    return {"mode": "local_debug", "certificate_thumbprint": thumbprint}


if __name__ == "__main__":
    try:
        if os.name != "nt":
            raise RuntimeError("Run certificate setup on the Windows computer used for building.")
        if sys.argv[1:] != ["setup"]:
            raise RuntimeError("Usage: py -3.12 tools\\debug_signing.py setup")
        setup()
    except (OSError, RuntimeError, ValueError, EOFError, subprocess.SubprocessError) as error:
        print("SIGNING SETUP FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
