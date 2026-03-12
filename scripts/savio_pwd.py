#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import struct
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Savio/BRC password as PIN+TOTP without storing secrets in git.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional local env file, e.g. .secrets/savio.env",
    )
    parser.add_argument("--pin", type=str, default=None, help="PIN override.")
    parser.add_argument("--otp-uri", type=str, default=None, help="otpauth:// URI override.")
    parser.add_argument("--secret", type=str, default=None, help="Base32 secret override.")
    parser.add_argument("--digits", type=int, default=None, help="OTP digits override.")
    parser.add_argument("--period", type=int, default=None, help="OTP period override.")
    parser.add_argument("--algorithm", type=str, default=None, help="OTP algorithm override.")
    parser.add_argument(
        "--otp-only",
        action="store_true",
        help="Print only the current OTP instead of PIN+OTP.",
    )
    parser.add_argument(
        "--seconds-left",
        action="store_true",
        help="Also print remaining seconds before the OTP rotates.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_local_config_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path
    env_path = os.getenv("SAVIO_PWD_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    repo_default = Path(".secrets/savio.env")
    return repo_default if repo_default.exists() else None


def get_value(
    cli_value: str | int | None,
    env_key: str,
    file_values: dict[str, str],
    default: str | int | None = None,
) -> str | int | None:
    if cli_value is not None:
        return cli_value
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    if env_key in file_values:
        return file_values[env_key]
    return default


def parse_otpauth(uri: str) -> dict[str, str | int]:
    parsed = urlparse(uri)
    if parsed.scheme != "otpauth":
        raise ValueError("OTP URI must start with otpauth://")
    query = parse_qs(parsed.query)
    return {
        "secret": query.get("secret", [None])[0],
        "algorithm": query.get("algorithm", ["SHA1"])[0],
        "digits": int(query.get("digits", [6])[0]),
        "period": int(query.get("period", [30])[0]),
    }


def base32_decode(secret: str) -> bytes:
    cleaned = secret.upper().strip().replace(" ", "").replace("-", "")
    cleaned += "=" * ((-len(cleaned)) % 8)
    return base64.b32decode(cleaned)


def get_totp(secret: str, digits: int, period: int, algorithm: str) -> str:
    algo = {
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }.get(algorithm.lower())
    if algo is None:
        raise ValueError(f"Unsupported TOTP algorithm: {algorithm}")
    key = base32_decode(secret)
    counter = int(time.time()) // period
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, algo).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10**digits)).zfill(digits)


def main() -> int:
    args = parse_args()
    config_path = resolve_local_config_path(args.config)
    file_values = load_env_file(config_path) if config_path is not None else {}

    pin = get_value(args.pin, "SAVIO_PIN", file_values)
    otp_uri = get_value(args.otp_uri, "SAVIO_OTP_URI", file_values)
    secret = get_value(args.secret, "SAVIO_OTP_SECRET", file_values)
    digits_value = get_value(args.digits, "SAVIO_OTP_DIGITS", file_values, 6)
    period_value = get_value(args.period, "SAVIO_OTP_PERIOD", file_values, 30)
    algorithm = str(get_value(args.algorithm, "SAVIO_OTP_ALGORITHM", file_values, "SHA1"))

    digits = int(digits_value) if digits_value is not None else 6
    period = int(period_value) if period_value is not None else 30

    if otp_uri:
        parsed = parse_otpauth(str(otp_uri))
        secret = secret or parsed["secret"]
        digits = int(parsed["digits"])
        period = int(parsed["period"])
        algorithm = str(parsed["algorithm"])

    if not secret:
        print("Missing TOTP secret. Set SAVIO_OTP_URI or SAVIO_OTP_SECRET.", file=sys.stderr)
        return 2
    if not args.otp_only and not pin:
        print("Missing PIN. Set SAVIO_PIN or use --otp-only.", file=sys.stderr)
        return 2

    otp = get_totp(str(secret), digits, period, algorithm)
    if args.seconds_left:
        seconds_left = period - (int(time.time()) % period)
        print(seconds_left, file=sys.stderr)

    if args.otp_only:
        print(otp)
    else:
        print(f"{pin}{otp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
