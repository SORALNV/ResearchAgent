# ResearchAgent 戦略ロードマップ

最終更新: 2026-07-10（WP14実装と同時に作成。以後、大きな設計判断のたびに更新する）

## 1. ビジョン

**「一人の優秀な研究員」をDiscordの向こうに置く。**
テーマを壁打ちで固めると、自分で実験計画を立て、実験し、失敗したら冷静に原因分析して別ルートを試し、証拠に基づいて完了を判定し、査読を通ったレポートを提出する。ユーザーの介入は「方向の決定」と「危険操作の承認」だけに絞る。

## 2. 現在のアーキテクチャ（2026-07-10時点）

```
/re plan ──► PlanningDialogue（codexスレッド継続・SEARCH_QUERY駆動の文献調査）
                │  PLAN_UPDATE: GOAL_CRITERIA / METHOD / CONSTRAINT / PROJECT_NAME
                ▼
/re start ──► ExperimentRoadmapPlanner（初回のみ。planned実験を種まき）
                │
                ▼
   ┌── 研究ラウンド（SubAgentCommandRunner: codex workspace-write）
   │      ・バックログ優先度順に実験、verify-before-claim、失敗時ラウンド内リトライ
   │      ▼
   │   RoundDirector（read-only 1発呼び）
   │      ・criteria照合（証拠ゲート）・DONEゲート・FAILURE_ANALYSIS（省察）
   │      ・EXPERIMENTS_UPDATE（バックログ手入れ）・停滞時は方針転換指示
   │      ▼
   │   ハーネス側の機械ゲート
   │      ・evidence空のmet→partial降格 ・停滞検知（STALL_ROUNDS）
   │      ・DONE前に ReportReviewer 査読（ACCEPT/REVISE）
   └──◄ CONTINUE（next_task注入） / FINAL_REPORT / DONE / BLOCKED

/re fork Vxxx ──► 続きバージョン（V009.0→V009.1）。criteria・教訓・バックログ・文献を継承
/re stop ──► harness_report.md / run_summary.md / INDEX.md / agent_memoryサマリ
```

状態はすべてファイル（state.json / journal.jsonl / research_ledger.jsonl / experiments.jsonl / papers.jsonl / plan.md）。プロセスを跨いで再開可能。

**モデル階層**（.envで変更可）: PLAN=5.4mini（安い・対話）、RESEARCH=gpt-5.5（実験実行）、DIRECTOR/REVIEWER=既定でPLANモデル（検証は安く高頻度に）、ROADMAP=既定でRESEARCHモデル。

## 3. 設計原則（これまでの学びの蒸留）

1. **プロンプトのお願いより、ハーネスの機械ゲート。** 「証拠を書け」と頼むより、evidence空のmetをharness側でpartialに降格する方が確実（WP12の証拠ゲート、WP13の停滞検知・査読ゲート）。
2. **1発呼び（ephemeral codex exec）の組み合わせで作る。** 長寿命エージェントより、役割ごとの短い呼び出し＋ファイル状態の方がデバッグ・再開・テストが容易。例外はPLAN対話（スレッド継続が必須と実証済み・WP4）。
3. **失敗は資産。** 失敗→FAILURE_ANALYSIS→lessons→全プロンプトに再注入→agent_memoryで次の研究へ（Reflexion/Voyagerの翻訳）。
4. **完了は疑ってかかる。** DONE主張はcriteria未達なら棄却、レポートは査読を通す（AI Scientistの査読ゲートの翻訳）。
5. **テストは実LLMを呼ばない。** ディレクター/査読者/ロードマップはすべて注入可能なスタブで検証。サブプロセス境界だけ寛容パーサ＋fallback。

## 4. ロードマップ

### 短期（次の1〜2週間）: 実機検証と足場固め
- [ ] **実機検証ラン**: WP10〜14の全機構を実際の研究テーマで通す（V009の続き＝`/re fork V009` が最適。自然動画5〜10本でのGemini埋め込み比較）。見どころ: ロードマップ播種→バックログ消化→証拠ゲート→（起きれば）停滞pivot→査読→INDEX.md系譜
- [ ] 検証で出た不具合の修正（プロンプト調整・パーサ強化を想定）
- [ ] Discord Botトークンの再発行（旧codexログに平文残存。既知の未対応セキュリティ課題）
- [ ] `/re status` に実験バックログ残数・停滞カウンタも表示（運用時の可観測性）

### 中期（1〜2ヶ月）: 研究の質と自己評価
- [ ] **小型RE-Bench化（自前評価ハーネス）**: 「答えを知っている小さな研究課題」を5〜10本用意し、隠し採点器（rubric＋時間/トークン予算）で自動採点する回帰テスト。ハーネス改修のたびに研究品質の退行を検知する（MLE-bench=75件のKaggle系実装課題、RE-Bench=時間予算別進捗、PaperBench=8,316個の階層rubricの縮小翻訳）
- [ ] **結果レジストリの中核化**: レポート・論文の全数値・図表・主張を「実験ログ由来」に限定する仕組み。experiments.jsonlのmetricsを唯一の数値ソースとし、レポート内の数値をログと機械照合（現在の査読ゲートを一段強化。AI Scientist v2は表の値をログから自動生成する）
- [ ] **論文パイプライン強化**: /re paper を査読ゲートと統合（thesis.mdにもReportReviewer相当）。引用のDOI/arXiv実在検証（P-xxxチェックの外部照合版）、図表とcaptionの整合チェック
- [ ] **仮説の分岐探索（小規模）**: バックログを線形キューから「仮説枝＋証拠価値＋残予算」の軽い木構造に拡張。Jetson 1台なのでGPU並列はせず、2〜3枝を小予算プロキシ実験で時分割し、進捗価値/残予算で枝刈りする（AI Scientist v2のprogressive tree searchの最小翻訳）
- [ ] **コスト計測の実質化と事前予算**: codex execのトークン/時間をjournalに記録し /re cost で見える化。さらに「実行前に最大retry・最大呼び出し数・モデル割当を確定し、逸脱で停止」する事前予算方式へ（後から請求を見るのでは遅い）

### 長期（3ヶ月〜）: スケールと自己改善
- [ ] **記憶の型分けとgain計測**: agent_memoryを「成功skill / 失敗pattern / 環境fact / 評価metric」の4型に分け、A-MEM流に記憶同士をリンク。ただしCL-Benchは「専用記憶が単純な文脈注入に負ける」例も示すため、**記憶の再利用が実際に成果を改善したかを計測してから拡張する**（増やすこと自体を目的にしない）
- [ ] **HITLの承認カード化**: 人間チェックポイントを「人間にしか判断できない分岐」に限定して整理 — 研究目的確定（済: PLAN）、仮説採択、高コスト本実験の直前、結論主張前（済: 査読）、公開/保存前。Discordに差分要約つき承認カードを出す形式へ
- [ ] **マルチプロジェクト並行運用**: 現在はアクティブセッション1つ。プロジェクトごとにチャンネル分離 or セッション切り替えコマンド
- [ ] **成果の相互参照**: 完了レポートをプロジェクト横断で検索し、新PLAN時に「過去の自分の研究」を文献同様に引用（AgentRxivの単機版）
- [ ] **ハーネス自己チューニング**: 蓄積したledger/journalから「どのゲートが何回発動したか」を集計し、閾値（STALL_ROUNDS等）の推奨値を自己提案

### 検討したが当面やらないこと
- 常駐マルチエージェント化（1発呼び原則を崩すほどの利益が現状ない）
- クラウドGPU等の重い実験基盤（承認ゲート越しの長時間ジョブpaternで足りている）
- 完全自動の論文投稿（人間の最終確認を外す理由がない）

## 5. リスクと運用課題

| リスク | 現状の手当て | 残課題 |
|---|---|---|
| Botトークン漏洩 | 再発行を推奨済み | **未対応。最優先** |
| コスト暴走 | MAX_ROUNDS/MAX_AGENT_CALLS/承認ゲート | トークン実測なし（中期で対応） |
| 査読・ディレクターの共倒れ | fallback=ACCEPT/CONTINUE（完了をブロックしない） | fallback多発の検知アラートなし |
| 停滞の偽検知 | met数とdone実験数の2指標 | 長い1実験（複数ラウンド）で誤検知の可能性。実機で観察 |
| ディスク肥大 | run_summary/INDEXで俯瞰可能 | 古いバージョンのアーカイブ方針未定 |

## 6. 参照（設計の出自）

- WP4: PLAN対話のスレッド継続化 / WP5-7: 実験管理・論文・長期記憶 / WP10: ディレクター検証 / WP11: PLAN構造化 / WP12: バックログ駆動+証拠ゲート / WP13: 省察・停滞・査読 / WP14: バージョン系譜・fork・レポート刷新
- 参考OSS: The AI Scientist（査読ゲート・buggyノード分類）、Reflexion(省察の再注入)、Agent Laboratory（エラー履歴注入・mle-solver）、Voyager（スキルライブラリ）、OpenHands/SWE-agent（StuckDetector・再問い合わせ上限）、Curie（実験の厳密性エンジン）、FableCodex（証拠必須チェックポイント）、Agent-Reach（失敗時の別ルート）
- 2026-07-10調査（codex web survey）の主要出典:
  - 評価: [MLE-bench](https://arxiv.org/abs/2410.07095) / [RE-Bench](https://arxiv.org/abs/2411.15114) / [PaperBench](https://arxiv.org/abs/2504.01848)
  - 木探索: [AI Scientist v2](https://arxiv.org/abs/2504.08066) / [BAVT](https://arxiv.org/html/2603.12634)
  - 論文執筆・HITL: [Agent Laboratory](https://arxiv.org/abs/2501.04227) / [Claw AI Lab](https://arxiv.org/html/2605.22662v1)
  - 記憶: [A-MEM](https://arxiv.org/abs/2502.12110) / [CL-Bench](https://arxiv.org/abs/2606.05661)（記憶は再利用gainを測ってから拡張、の根拠）

## 7. 調査が示した推奨優先順位（2026-07-10時点の判断）

1. **小型RE-Bench化** — ハーネスの改善が研究品質に効いているかを測れないと、以降の全改善が勘になる
2. **実験木探索**（線形バックログ→仮説枝＋証拠価値＋残予算）
3. **結果レジストリ中核化** — レポートの全数値をログ由来に限定
4. **記憶の型分けと再利用gain計測**
5. **Discord HITL承認カード化** — 本実験前・枝刈り前・結論採択前のみ人間確認

ただし最優先は上記より先に「**実機検証ラン**」。WP10〜14は全てユニットテスト検証のみで、実際の研究ランでの動作は未確認（2026-07-10時点）。
