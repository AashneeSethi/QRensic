"""
QRensic - Amazon Bedrock Runtime Smoke Test Script
--------------------------------------------------
Verifies connectivity, IAM permissions, and model access to Amazon Bedrock Runtime
using the Converse API.

Usage:
  python scripts/bedrock_smoke_test.py [--model-id <id>] [--region <region>]

Environment Variables:
  AWS_REGION          AWS region (default: us-east-1)
  BEDROCK_MODEL_ID    Model ID (default: anthropic.claude-3-haiku-20240307-v1:0)
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


def run_smoke_test(model_id: str, region: str) -> int:
    print("\n========================================================")
    print("  Q R E N S I C   B E D R O C K   S M O K E   T E S T")
    print("========================================================")
    print(f"Target Region   : {region}")
    print(f"Target Model ID : {model_id}")
    print("--------------------------------------------------------")

    # Step 1: Check AWS credential resolution
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()

    if credentials is None:
        print("[FAIL] AWS Credentials Not Detected.")
        print("\nActionable Setup Steps:")
        print("  1. Configure credentials via AWS CLI:")
        print("       aws configure")
        print("  2. Or set standard environment variables in your terminal:")
        print("       Windows PowerShell:")
        print("         $env:AWS_ACCESS_KEY_ID = \"your_access_key_id\"")
        print("         $env:AWS_SECRET_ACCESS_KEY = \"your_secret_access_key\"")
        print("         $env:AWS_REGION = \"us-east-1\"")
        print("         (Optional STS) $env:AWS_SESSION_TOKEN = \"your_session_token\"")
        print("       Linux / macOS / Bash:")
        print("         export AWS_ACCESS_KEY_ID=\"your_access_key_id\"")
        print("         export AWS_SECRET_ACCESS_KEY=\"your_secret_access_key\"")
        print("         export AWS_REGION=\"us-east-1\"")
        print("  3. Verify your IAM identity has 'bedrock:InvokeModel' permission.")
        print("========================================================\n")
        return 1

    # Step 2: Initialize client with strict timeout
    try:
        config = Config(
            connect_timeout=5,
            read_timeout=15,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        client = session.client("bedrock-runtime", region_name=region, config=config)
    except Exception as e:
        print(f"[FAIL] Failed to construct Bedrock Runtime client: {e}")
        return 1

    # Step 3: Minimal Converse API probe
    probe_messages = [
        {
            "role": "user",
            "content": [{"text": "QRensic probe test. Reply with: OK"}],
        }
    ]

    try:
        response = client.converse(
            modelId=model_id,
            messages=probe_messages,
            inferenceConfig={
                "maxTokens": 20,
                "temperature": 0.0,
            },
        )
    except NoCredentialsError:
        print("[FAIL] No credentials found by boto3 credential provider chain.")
        print("Actionable: Ensure credentials are in ~/.aws/credentials or environment variables.")
        return 1
    except PartialCredentialsError:
        print("[FAIL] Incomplete credentials found (missing secret key or access key).")
        print("Actionable: Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are both set.")
        return 1
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "ClientError")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"[FAIL] Bedrock ClientError [{error_code}]")
        print(f"Message: {error_msg}\n")

        if error_code == "AccessDeniedException":
            print("Actionable Steps for AccessDeniedException:")
            print("  1. Model Access: In the AWS Console, navigate to Amazon Bedrock -> Model access.")
            print(f"     Ensure '{model_id}' is enabled/granted for your account in region '{region}'.")
            print("  2. IAM Permissions: Ensure your IAM User/Role policy contains:")
            print("       {")
            print("         \"Effect\": \"Allow\",")
            print("         \"Action\": [\"bedrock:InvokeModel\", \"bedrock:InvokeModelWithResponseStream\"],")
            print("         \"Resource\": \"*\"")
            print("       }")
        elif error_code == "ResourceNotFoundException":
            print("Actionable Steps for ResourceNotFoundException:")
            print(f"  Model ID '{model_id}' was not found in region '{region}'.")
            print("  Verify model availability in the AWS Bedrock console for this region.")
            print("  Try an alternative enabled model, e.g. amazon.nova-micro-v1:0 or anthropic.claude-3-5-sonnet-20240620-v1:0.")
        elif error_code == "ValidationException":
            print("Actionable Steps for ValidationException:")
            print(f"  The model '{model_id}' may not support the Bedrock Converse API.")
            print("  Use a Converse-compatible foundation model (Claude 3 / 3.5 or Amazon Nova).")
        elif error_code in ("ThrottlingException", "TooManyRequestsException"):
            print("Actionable: Request rate exceeded. Bedrock Runtime quota was reached. Retry shortly.")
        else:
            print(f"Actionable: Check AWS Bedrock console service status and IAM permissions in region '{region}'.")

        print("========================================================\n")
        return 1
    except BotoCoreError as e:
        print(f"[FAIL] Transport/Network error: {e}")
        print(f"Actionable: Check internet connection and verify region '{region}' is accessible.")
        print("========================================================\n")
        return 1
    except Exception as e:
        print(f"[FAIL] Unexpected error during Converse call: {e}")
        print("========================================================\n")
        return 1

    # Step 4: Validate successful response
    output_message = response.get("output", {}).get("message", {})
    content = output_message.get("content", [])
    response_text = " ".join([b.get("text", "") for b in content if "text" in b]).strip()
    stop_reason = response.get("stopReason", "unknown")
    usage = response.get("usage", {})

    print("[SUCCESS] Amazon Bedrock Converse probe succeeded!")
    print(f"Model ID        : {model_id}")
    print(f"Region          : {region}")
    print(f"Stop Reason     : {stop_reason}")
    print(f"Tokens Used     : Input={usage.get('inputTokens', 0)}, Output={usage.get('outputTokens', 0)}")
    print(f"Model Response  : \"{response_text}\"")
    print("========================================================\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QRensic - Amazon Bedrock Runtime Smoke Test Script"
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        help=f"Amazon Bedrock Model ID (default: env BEDROCK_MODEL_ID or {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default=os.environ.get("AWS_REGION", DEFAULT_REGION),
        help=f"AWS Region (default: env AWS_REGION or {DEFAULT_REGION})",
    )
    args = parser.parse_args()
    return run_smoke_test(model_id=args.model_id, region=args.region)


if __name__ == "__main__":
    sys.exit(main())
