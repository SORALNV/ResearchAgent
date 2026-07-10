# GCP 利用時の確認手順（ResearchAgent）

このリポジトリでは、`~/.config/gcloud` の書き込み権限に依存しない運用を想定し、以下を使ってください。

## 1) 端末で一度読み込み

```bash
source scripts/gcp_env.sh
```

## 2) プロジェクトを切り替え

```bash
gcp_use_plan_project embedding-2-490304
```

必要なら将来のテーマごとに上記だけを実行して `PROJECT_ID` を変更します。

### Plan会話で受け取った場合

このリポジトリでは、次のいずれかを受け取ると、同一コマンドで反映できます。

- `GOOGLE_CLOUD_PLAN_PROJECT_ID` 環境変数
- `RESEARCH_AGENT_PROJECT_ID` 環境変数
- `RESEARCH_AGENT_PLAN_PROJECT_ID` 環境変数

例:

```bash
export GOOGLE_CLOUD_PLAN_PROJECT_ID=embedding-2-490304
gcp_apply_plan
```

## 3) 事前確認

```bash
gcp_status
```

確認ポイント:
- `core/project` が現在のテーマの `PROJECT_ID`
- `aiplatform.googleapis.com` が有効
- ADC ファイル内の `quota_project_id` がプロジェクトID

## 4) API/権限の確認（ネットワーク必要）

```bash
gcloud services list --project "$GOOGLE_CLOUD_PROJECT" --enabled --filter="config.name:aiplatform.googleapis.com" --format="value(config.name)"
gcloud beta billing projects describe "$GOOGLE_CLOUD_PROJECT" --format="value(billingEnabled)"
```

`auth list` で有効アカウント、`roles/owner` もしくは `roles/aiplatform.user` 以上があればOKです。
