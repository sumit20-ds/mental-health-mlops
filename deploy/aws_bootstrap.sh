#!/usr/bin/env bash
# One-time AWS setup: ECR repo, S3 bucket, SageMaker execution role, GitHub OIDC deploy role.
# Usage:  GITHUB_REPO=<your-gh-user>/student-mental-health-mlops AWS_REGION=us-east-1 bash deploy/aws_bootstrap.sh
set -euo pipefail

: "${GITHUB_REPO:?set GITHUB_REPO=<owner>/<repo>}"
REGION="${AWS_REGION:-us-east-1}"
PROJECT="student-mental-health"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${PROJECT}-mlops-${ACCOUNT}-${REGION}"
REPO="${PROJECT}-api"
EXEC_ROLE="${PROJECT}-sagemaker-exec"
DEPLOY_ROLE="${PROJECT}-github-deploy"

echo ">> ECR repository"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null

echo ">> S3 bucket (model artifacts + data capture)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo ">> SageMaker execution role"
cat > /tmp/sm-trust.json <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$EXEC_ROLE" --assume-role-policy-document file:///tmp/sm-trust.json >/dev/null
cat > /tmp/sm-exec-policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer","ecr:BatchCheckLayerAvailability"],
  "Resource":"arn:aws:ecr:${REGION}:${ACCOUNT}:repository/${REPO}"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::${BUCKET}","arn:aws:s3:::${BUCKET}/*"]},
 {"Effect":"Allow","Action":["s3:PutObject"],"Resource":"arn:aws:s3:::${BUCKET}/data-capture/*"},
 {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}
JSON
aws iam put-role-policy --role-name "$EXEC_ROLE" --policy-name inference --policy-document file:///tmp/sm-exec-policy.json
EXEC_ARN="arn:aws:iam::${ACCOUNT}:role/${EXEC_ROLE}"

echo ">> GitHub OIDC provider + deploy role (no long-lived AWS keys in GitHub)"
OIDC_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"
aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1 || \
  aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 >/dev/null
cat > /tmp/gh-trust.json <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Federated":"${OIDC_ARN}"},
 "Action":"sts:AssumeRoleWithWebIdentity",
 "Condition":{"StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com"},
              "StringLike":{"token.actions.githubusercontent.com:sub":"repo:${GITHUB_REPO}:*"}}}]}
JSON
if aws iam get-role --role-name "$DEPLOY_ROLE" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$DEPLOY_ROLE" --policy-document file:///tmp/gh-trust.json
else
  aws iam create-role --role-name "$DEPLOY_ROLE" --assume-role-policy-document file:///tmp/gh-trust.json >/dev/null
fi
cat > /tmp/gh-deploy-policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
  "ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],
  "Resource":"arn:aws:ecr:${REGION}:${ACCOUNT}:repository/${REPO}"},
 {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::${BUCKET}","arn:aws:s3:::${BUCKET}/*"]},
 {"Effect":"Allow","Action":["sagemaker:CreateModel","sagemaker:CreateEndpointConfig","sagemaker:CreateEndpoint",
  "sagemaker:UpdateEndpoint","sagemaker:DescribeEndpoint","sagemaker:DescribeEndpointConfig","sagemaker:InvokeEndpoint"],"Resource":"*"},
 {"Effect":"Allow","Action":"iam:PassRole","Resource":"${EXEC_ARN}"}]}
JSON
aws iam put-role-policy --role-name "$DEPLOY_ROLE" --policy-name deploy --policy-document file:///tmp/gh-deploy-policy.json

cat <<OUT

=====================  DONE - add these in GitHub -> Settings -> Secrets and variables -> Actions  =====================
Repository VARIABLES:
  AWS_REGION                = ${REGION}
  ECR_REPOSITORY            = ${REPO}
  S3_BUCKET                 = ${BUCKET}
  SAGEMAKER_ENDPOINT_NAME   = ${PROJECT}-endpoint
Repository SECRETS:
  AWS_DEPLOY_ROLE_ARN       = arn:aws:iam::${ACCOUNT}:role/${DEPLOY_ROLE}
  SAGEMAKER_ROLE_ARN        = ${EXEC_ARN}
=======================================================================================================================
OUT
