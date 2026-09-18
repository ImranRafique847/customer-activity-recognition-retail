"""
scripts/launch_training.py
==========================
Launches an EC2 g4dn.xlarge instance with the correct Deep Learning AMI,
uploads the latest project files, and provides the connect command.

Usage:
    py -3.11 scripts/launch_training.py              # launch + connect info
    py -3.11 scripts/launch_training.py --terminate  # terminate training instance
"""

import argparse
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

REGION          = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
AMI_ID          = "ami-0016081b488c7376d"   # Deep Learning OSS PyTorch 2.7 Ubuntu 22.04
INSTANCE_TYPE   = "g4dn.xlarge"
KEY_NAME        = "retail-cv-key"
SG_NAME         = "retail-cv-training-sg"
IAM_PROFILE     = "EC2-RetailCV-Role"
INSTANCE_TAG    = "retail-cv-training"
BUCKET          = "retaildata-cv-2026"
VOLUME_SIZE_GB  = 150

# Files to upload before training
FILES_TO_UPLOAD = [
    ("train_carr.py",                          "project/train_carr.py"),
    ("balance_dataset.py",                     "project/balance_dataset.py"),
    ("models/__init__.py",                     "project/models/__init__.py"),
    ("models/cbam.py",                         "project/models/cbam.py"),
    ("models/register.py",                     "project/models/register.py"),
    ("configs/yolo11s-cbam.yaml",              "project/configs/yolo11s-cbam.yaml"),
    ("data/retail_cv_balanced/dataset.yaml",   "project/data/retail_cv_balanced/dataset.yaml"),
    ("data/retail_cv_balanced/train.txt",      "project/data/retail_cv_balanced/train.txt"),
    ("data/retail_cv_balanced/val.txt",        "project/data/retail_cv_balanced/val.txt"),
    ("data/retail_cv_balanced/test.txt",       "project/data/retail_cv_balanced/test.txt"),
]

# Startup script that runs automatically when EC2 boots
USER_DATA = """#!/bin/bash
export PATH=/opt/pytorch/bin:$PATH
echo 'export PATH=/opt/pytorch/bin:$PATH' >> /home/ubuntu/.bashrc

# Install packages
/opt/pytorch/bin/pip install ultralytics python-dotenv boto3 -q

# Download project files
mkdir -p /home/ubuntu/retail-cv
aws s3 sync s3://{bucket}/project/ /home/ubuntu/retail-cv/ --region {region}

# Download dataset
aws s3 sync s3://{bucket}/extracted_images_and_labels/data/ /home/ubuntu/retail-cv/data/s3_cache/ --region {region}

echo "SETUP COMPLETE" > /home/ubuntu/setup_done.txt
""".format(bucket=BUCKET, region=REGION)


def get_ec2_client():
    return boto3.client(
        "ec2",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=REGION,
    )


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=REGION,
    )


def upload_project_files():
    """Upload latest project files to S3 before launching EC2."""
    s3 = get_s3_client()
    print("Uploading project files to S3...")
    for local, s3_key in FILES_TO_UPLOAD:
        if Path(local).exists():
            s3.upload_file(local, BUCKET, s3_key)
            print(f"  ✅ {local}")
        else:
            print(f"  ⚠️  MISSING: {local}")
    print()


def get_or_create_security_group(ec2) -> str:
    try:
        sgs = ec2.describe_security_groups(GroupNames=[SG_NAME])
        sg_id = sgs["SecurityGroups"][0]["GroupId"]
        print(f"Using existing security group: {sg_id}")
        return sg_id
    except Exception:
        sg = ec2.create_security_group(
            GroupName=SG_NAME,
            Description="Retail CV training security group",
        )
        sg_id = sg["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
            }],
        )
        print(f"Created security group: {sg_id}")
        return sg_id


def launch_instance(ec2) -> dict:
    sg_id = get_or_create_security_group(ec2)

    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=USER_DATA,
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": VOLUME_SIZE_GB, "VolumeType": "gp3"},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": INSTANCE_TAG},
                {"Key": "Project", "Value": "retail-cv-analytics"},
            ],
        }],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Instance launched: {instance_id}")
    print("Waiting for public IP (30s)...")
    time.sleep(30)

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    return inst


def terminate_instance(ec2):
    r = ec2.describe_instances(
        Filters=[{"Name": "tag:Name", "Values": [INSTANCE_TAG]},
                 {"Name": "instance-state-name", "Values": ["running", "stopped"]}]
    )
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            iid = inst["InstanceId"]
            ec2.terminate_instances(InstanceIds=[iid])
            print(f"Terminated: {iid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminate", action="store_true",
                        help="Terminate the training instance")
    args = parser.parse_args()

    ec2 = get_ec2_client()

    if args.terminate:
        terminate_instance(ec2)
        return

    # Upload latest files
    upload_project_files()

    # Launch instance
    inst = launch_instance(ec2)
    ip   = inst.get("PublicIpAddress", "pending")
    iid  = inst["InstanceId"]

    print(f"\n{'='*60}")
    print(f" EC2 Instance Ready")
    print(f"{'='*60}")
    print(f" Instance ID: {iid}")
    print(f" Public IP:   {ip}")
    print(f" State:       {inst['State']['Name']}")
    print(f"\n NOTE: UserData script is running in background:")
    print(f"   - Installing packages")
    print(f"   - Downloading project files from S3")
    print(f"   - Downloading dataset from S3 (~18GB, 30-60 min)")
    print(f"\n Connect via EC2 Instance Connect in AWS Console")
    print(f" Or SSH: ssh -i ~/.ssh/retail-cv-key.pem ubuntu@{ip}")
    print(f"\n Check setup progress:")
    print(f"   cat /home/ubuntu/setup_done.txt")
    print(f"\n When setup is done, run CBAM training:")
    print(f"   tmux new -s training")
    print(f"   cd ~/retail-cv && python balance_dataset.py --max-per-class 5000")
    print(f"   python -c \"")
    print(f"from models.register import register_custom_modules")
    print(f"register_custom_modules()")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('configs/yolo11s-cbam.yaml')")
    print(f"model.train(")
    print(f"    data='data/retail_cv_balanced/dataset.yaml',")
    print(f"    epochs=200, batch=32, workers=4, imgsz=640,")
    print(f"    device=0, project='runs/carr', name='cbam_5k',")
    print(f"    optimizer='AdamW', lr0=0.001, lrf=0.01,")
    print(f"    weight_decay=0.0005, warmup_epochs=5,")
    print(f"    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,")
    print(f"    degrees=10.0, translate=0.1, scale=0.5,")
    print(f"    shear=2.0, perspective=0.0005,")
    print(f"    flipud=0.3, fliplr=0.5,")
    print(f"    mosaic=1.0, mixup=0.15, copy_paste=0.1,")
    print(f"    close_mosaic=20, patience=50, plots=True, amp=True")
    print(f")\" 2>&1 | tee cbam_5k.log")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
