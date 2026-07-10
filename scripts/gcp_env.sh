#!/usr/bin/env bash
# Shell helper for Google Cloud work in this repository.
# Usage:
#   source scripts/gcp_env.sh
#   gcp_set_project YOUR_PROJECT_ID
#   gcp_use_plan_project [PROJECT_ID]

: "${GCLOUD_WORKDIR:=$HOME/.gcloud-researchagent}"
: "${GCLOUD_PLAN_PROJECT_FILE:=$GCLOUD_WORKDIR/last_project_id}"

export CLOUDSDK_CONFIG="$GCLOUD_WORKDIR"
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE:-$HOME/.config/gcloud/application_default_credentials.json}"

mkdir -p "$CLOUDSDK_CONFIG"
if [ -d "$CLOUDSDK_CONFIG" ]; then
  chmod 700 "$CLOUDSDK_CONFIG" 2>/dev/null || true
fi

gcp_set_project() {
  local project_id="$1"
  if [ -z "${project_id:-}" ]; then
    echo "Usage: gcp_set_project <PROJECT_ID>"
    return 1
  fi

  export RESEARCH_AGENT_PROJECT_ID="$project_id"
  export GOOGLE_CLOUD_PROJECT="$project_id"

  gcloud config set project "$project_id"
  gcloud auth application-default set-quota-project "$project_id"
}

_gcp_candidate_project() {
  local project_id="${1:-}"
  if [ -n "${project_id:-}" ]; then
    echo "$project_id"
    return 0
  fi

  if [ -n "${GOOGLE_CLOUD_PLAN_PROJECT_ID:-}" ]; then
    echo "$GOOGLE_CLOUD_PLAN_PROJECT_ID"
    return 0
  fi

  if [ -n "${RESEARCH_AGENT_PROJECT_ID:-}" ]; then
    echo "$RESEARCH_AGENT_PROJECT_ID"
    return 0
  fi

  if [ -n "${RESEARCH_AGENT_PLAN_PROJECT_ID:-}" ]; then
    echo "$RESEARCH_AGENT_PLAN_PROJECT_ID"
    return 0
  fi

  if [ -f "$GCLOUD_PLAN_PROJECT_FILE" ]; then
    cat "$GCLOUD_PLAN_PROJECT_FILE"
    return 0
  fi

  return 1
}

gcp_use_plan_project() {
  local project_id
  project_id=$(_gcp_candidate_project "$1")
  if [ -z "$project_id" ]; then
    echo "No project id was provided. Set one of:"
    echo "  - PROJECT_ID argument"
    echo "  - GOOGLE_CLOUD_PLAN_PROJECT_ID environment variable"
    echo "  - RESEARCH_AGENT_PROJECT_ID environment variable"
    echo "  - RESEARCH_AGENT_PLAN_PROJECT_ID environment variable"
    echo "  - last used project: $GCLOUD_PLAN_PROJECT_FILE"
    return 1
  fi

  gcp_set_project "$project_id"
  echo "$project_id" > "$GCLOUD_PLAN_PROJECT_FILE"
  export RESEARCH_AGENT_PROJECT_ID="$project_id"
  echo "Project set for this session: $project_id"
}

gcp_apply_plan() {
  # Alias for the common flow in planning mode:
  # set project once from argument or env vars / saved value.
  gcp_use_plan_project "$1"
}

gcp_status() {
  echo "---- gcloud local config ----"
  gcloud config list core/project,account || true

  echo "---- ADC credential file ----"
  if [ -f "$CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE" ]; then
    grep -E "\"quota_project_id\"|\"client_id\"" "$CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE" || true
  else
    echo "ADC file not found: $CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"
  fi
}
