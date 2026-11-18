#!/bin/sh
# One AWS spot T4 (g4dn.xlarge) as an ephemeral GitHub Actions runner for the opt-in linux-gpu job, without RunsOn.
# Needs the IAM policy in doc/aws-t4-runner-policy.json on the signed-in role and `gh` with admin rights on the repository.
# See doc/ci-gpu-runners.md ("Own spot runner, no RunsOn").
#
#   sh test/gpuserv/t4_spot_runner.sh up [--dry-run]   launch one spot instance; it registers a one-job runner labelled $RUNNER_LABEL
#   sh test/gpuserv/t4_spot_runner.sh status           the instances tagged Project=pg-strom-ci and the repository's runners
#   sh test/gpuserv/t4_spot_runner.sh down             terminate every instance tagged Project=pg-strom-ci
#
# Then run the job on it:  gh workflow run vulkan-check.yml --ref vulkan-gpu -f run_gpu_t4=true -f gpu_runner_label=$RUNNER_LABEL
# The runner is ephemeral (takes one job, then the instance shuts itself down and is terminated) and the instance also powers off
# after MAX_MIN minutes whether or not a job came, so a forgotten instance costs at most MAX_MIN minutes of spot time.
#
# Environment: AWS_PROFILE (zrecore-ci), AWS_REGION (us-east-1), REPO (zrecore-org/pg-strom), RUNNER_LABEL (pgstrom-t4-aws),
# MAX_MIN (150), VOLUME_GB (100, the policy allows 150), AMI (default: newest AWS Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 24.04).
set -u
PROFILE=${AWS_PROFILE:-zrecore-ci}
REGION=${AWS_REGION:-us-east-1}
REPO=${REPO:-zrecore-org/pg-strom}
RUNNER_LABEL=${RUNNER_LABEL:-pgstrom-t4-aws}
MAX_MIN=${MAX_MIN:-150}
VOLUME_GB=${VOLUME_GB:-100}
TYPE=g4dn.xlarge
TAG=pg-strom-ci

aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
die() { echo "t4_spot_runner: $*" >&2; exit 1; }

live_instances() {
	aws_ ec2 describe-instances \
		--filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=pending,running" \
		--query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime,Placement.AvailabilityZone]' --output text
}

cmd_status() {
	echo "instances (tag Project=$TAG):"
	live_instances
	echo "runners of $REPO:"
	gh api "repos/$REPO/actions/runners" --jq '.runners[] | [.name, .status, (.busy|tostring), ([.labels[].name]|join(","))] | @tsv' 2>&1
}

cmd_down() {
	ids=$(live_instances | awk '{print $1}')
	[ -n "$ids" ] || { echo "no instance to terminate"; return 0; }
	# shellcheck disable=SC2086
	aws_ ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
}

cmd_up() {
	dry=
	[ "${1:-}" = "--dry-run" ] && dry=--dry-run
	command -v aws >/dev/null || die "aws CLI not found"
	[ -z "$dry" ] && { command -v gh >/dev/null || die "gh not found"; }

	# one runner at a time: the quota is 8 spot vCPUs shared with other work, and this keeps the bill predictable
	have=$(live_instances)
	[ -z "$have" ] || die "an instance is already running (use status / down):
$have"

	ami=${AMI:-}
	if [ -z "$ami" ]; then
		ami=$(aws_ ec2 describe-images --owners amazon --filters \
			'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*' 'Name=state,Values=available' \
			--query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text) || die "describe-images failed"
	fi
	case $ami in ami-*) ;; *) die "no AMI found ($ami)" ;; esac
	root=$(aws_ ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text) || die "AMI lookup failed"

	# the cheapest zone right now (spot price differs by zone), mapped to the default subnet of that zone
	az=$(aws_ ec2 describe-spot-price-history --instance-types $TYPE --product-descriptions Linux/UNIX \
		--start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --query 'SpotPriceHistory[].[SpotPrice,AvailabilityZone]' --output text |
		sort -n | head -n 1 | awk '{print $2}')
	[ -n "$az" ] || die "no spot price for $TYPE"
	subnet=$(aws_ ec2 describe-subnets --filters "Name=availability-zone,Values=$az" Name=default-for-az,Values=true \
		--query 'Subnets[0].SubnetId' --output text)
	case $subnet in subnet-*) ;; *) die "no default subnet in $az" ;; esac

	if [ -n "$dry" ]; then
		token=DRYRUN
	else
		token=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token) || die "registration token failed (needs admin on $REPO)"
		[ -n "$token" ] || die "empty registration token"
	fi

	ud=$(mktemp "${TMPDIR:-/tmp}/t4ud.XXXXXX") || die "mktemp"
	trap 'rm -f "$ud"' EXIT
	# user-data: runs as root at first boot. The token is single use and expires within an hour.
	cat > "$ud" <<EOF
#!/bin/bash
set -u
shutdown -h +$MAX_MIN
cat > /home/ubuntu/start-runner.sh <<'RUN'
#!/bin/bash
set -eu
cd /home/ubuntu
V=\$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -n 1)
curl -fsSL -o runner.tar.gz "https://github.com/actions/runner/releases/download/v\$V/actions-runner-linux-x64-\$V.tar.gz"
mkdir -p actions-runner && tar -xzf runner.tar.gz -C actions-runner && cd actions-runner
./config.sh --unattended --ephemeral --url https://github.com/$REPO --token $token --labels $RUNNER_LABEL --name t4-\$(date +%s) --replace
./run.sh || true
sudo shutdown -h now
RUN
chown ubuntu:ubuntu /home/ubuntu/start-runner.sh
su - ubuntu -c 'nohup bash /home/ubuntu/start-runner.sh > /home/ubuntu/runner.log 2>&1 &'
EOF

	echo "launching $TYPE spot in $az ($subnet), ami $ami, volume ${VOLUME_GB} GB, power-off after ${MAX_MIN} min"
	out=$(aws_ ec2 run-instances $dry --image-id "$ami" --instance-type $TYPE --count 1 --subnet-id "$subnet" \
		--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
		--instance-initiated-shutdown-behavior terminate \
		--metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
		--block-device-mappings "DeviceName=$root,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
		--tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$TAG},{Key=Name,Value=pgstrom-t4-runner}]" \
		"ResourceType=volume,Tags=[{Key=Project,Value=$TAG}]" \
		--user-data "file://$ud" --query 'Instances[0].InstanceId' --output text 2>&1)
	rc=$?
	if [ -n "$dry" ]; then
		case $out in
			*DryRunOperation*) echo "dry run OK: the policy allows the launch"; return 0 ;;
			*) echo "$out" | head -n 3; return 1 ;;
		esac
	fi
	[ $rc -eq 0 ] || die "run-instances failed: $out"
	echo "instance $out; the runner appears as '$RUNNER_LABEL' in a few minutes (status shows it)"
	echo "start the job:  gh workflow run vulkan-check.yml --ref vulkan-gpu -f run_gpu_t4=true -f gpu_runner_label=$RUNNER_LABEL"
}

case ${1:-} in
	up) shift; cmd_up "$@" ;;
	status) cmd_status ;;
	down) cmd_down ;;
	*) sed -n 2,14p "$0"; exit 2 ;;
esac
