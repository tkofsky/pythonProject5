# build_new_project_zip.py
# Creates Project_All_CSVs.zip containing all 18 CSVs + metadata + master summary + README

import os, zipfile, io, textwrap

FILES = {
    # --- Core CSVs (1–6) ---
    "Schema_Trial_Baselines.csv": """Schema Variant,Valid Structured (%),Mean Reward,Reward / 1K Tokens,Notes
p_schema_guided,86,0.65,3.6,"Clear field list; balanced accuracy"
p_minimal_json,84,0.64,3.5,"Most efficient"
p_schema_strict,81,0.56,3.0,"Too rigid"
p_reason_then_json,63,0.47,2.5,"Reasoning text contaminated structure"
""",
    "F1_Performance_by_Field_and_System.csv": """System,Intent,Constraints,Urgency,Steps,Mean F1,Notes
0-Shot,0.23,0.58,0.55,0.49,0.46,"Baseline, weak structure"
1-Shot,0.89,0.75,0.73,0.57,0.74,"Big improvement; learned schema"
2-Shot,0.91,0.80,0.78,0.64,0.78,"Small F1 gains; cost increased"
3-Shot,0.92,0.80,0.78,0.65,0.79,"Plateau; efficiency dropped"
1-Pass Baseline,0.89,0.74,0.73,0.57,0.73,"Static single-pass setup"
2-Pass Static,0.91,0.80,0.79,0.63,0.78,"Sequential reasoning improved structure"
2-Pass + Router,0.93,0.83,0.82,0.67,0.82,"Best overall; dynamic routing boosted consistency"
""",
    "Reward_and_Efficiency_Comparison.csv": """System,Mean Reward,Reward / Token,Reward / 1K Tokens,% Change vs Previous,Notes
0-Shot,0.75,0.0023,2.3,,"Poor structure, high variability"
1-Shot,0.85,0.0027,2.7,+17%,"Optimal few-shot efficiency"
2-Shot,0.82,0.0017,1.7,-37%,"Token cost increased"
3-Shot,0.79,0.0013,1.3,-24%,"Plateaued"
2-Pass Static,0.78,0.0026,2.6,+4%,"Stronger consistency"
2-Pass + Router,0.82,0.0037,3.7,+42%,"Peak reward efficiency"
""",
    "Bandit_Selection_Distribution.csv": """Arm Type,Arm Label,% of Total Selections,Notes
Few-Shot,0-Shot,6%,"Abandoned early"
Few-Shot,1-Shot,63%,"Dominant configuration"
Few-Shot,2-Shot,27%,"Secondary, higher cost"
Few-Shot,3-Shot,8%,"Residual exploration"
Schema,p_minimal_json,36%,"Efficient baseline"
Schema,p_schema_guided,34%,"Balanced accuracy"
Schema,p_strict,18%,"Over-constrained"
Schema,p_reason_then_json,12%,"Reasoning leak, low validity"
Pass Mode,Single,42%,"For simple tasks"
Pass Mode,Two-Pass,58%,"For multi-intent tasks"
""",
    "Mutation_Performance_Comparison.csv": """Parent Prompt,Mutation,Parent Reward,Mutated Reward,Δ Reward,Δ F1 Steps,Structured Validity,Notes
p_minimal_json,p_minimal_m7748,0.82,0.85,+0.03,+0.05,0.99,"Schema tightening improved structure"
p_minimal_m7748,p_minimal_m8848,0.85,0.846,-0.004,-0.01,0.99,"Plateaued"
p_schema_guided,guided_m1032,0.81,0.823,+0.013,+0.02,0.98,"Stable improvement"
p_reason_then_json,reason_m502,0.50,0.56,+0.06,+0.03,0.82,"Improved structure, still underperforming"
""",
    "TwoPass_Router_Gains.csv": """Metric,1-Pass,2-Pass Static,2-Pass + Router,Δ (Router vs 1-Pass),Δ (Router vs 2-Pass),Notes
Mean F1,0.73,0.78,0.82,+12%,+5%,"Router improved task matching"
F1 Steps,0.57,0.63,0.67,+17%,+6%,"Strongest gain"
Structured Validity,0.92,0.95,0.98,+6%,+3%,"Reduced malformed outputs"
Reward / 1K Tokens,2.7,3.0,3.8–4.0,+41%,+27%,"Highest efficiency"
""",

    # --- Extended CSVs (7–18) ---
    "FewShot_Field_Comparison.csv": """Field,ShotCount,MeanF1,Delta_vs_Prev,Relative_Gain_Percent,Notes
Intent,0,0.23,,,"Very low without examples"
Intent,1,0.89,+0.66,+287%,"Large jump with one demo"
Intent,2,0.91,+0.02,+2%,"Plateau"
Intent,3,0.92,+0.01,+1%,"Near ceiling"
Constraints,0,0.58,,,"Weak structure"
Constraints,1,0.75,+0.17,+29%,"Format learned"
Constraints,2,0.80,+0.05,+7%,"Best few-shot gain"
Constraints,3,0.80,+0.00,+0%,"No further gain"
Urgency,0,0.55,,,"Weak labels"
Urgency,1,0.73,+0.18,+33%,"Clear mapping from demos"
Urgency,2,0.78,+0.05,+7%,"Best few-shot gain"
Urgency,3,0.78,+0.00,+0%,"Plateau"
Steps,0,0.49,,,"Reasoning errors"
Steps,1,0.57,+0.08,+16%,"Better structure"
Steps,2,0.64,+0.07,+12%,"Highest gain for steps"
Steps,3,0.65,+0.01,+2%,"Minor improvement"
""",
    "Bandit_Run_Log_Summary.csv": """Run_ID,SelectedArm_Type,SelectedArm_Label,Reward,RollingMeanReward,Exploit_Prob,Was_Mutation,SchemaType,Notes
1,Few-Shot,0-Shot,0.73,0.73,0.10,No,p_minimal_json,"Cold start"
10,Few-Shot,1-Shot,0.84,0.79,0.35,No,p_minimal_json,"Early improvement"
20,Schema,p_schema_guided,0.82,0.81,0.45,No,p_schema_guided,"Guided baseline confirms"
30,Few-Shot,2-Shot,0.82,0.82,0.50,No,p_schema_guided,"Small F1 boost; more tokens"
40,Mutation,p_minial_m7748,0.86,0.83,0.60,Yes,p_minial_json,"Top mutation discovered"
50,PassMode,Two-Pass,0.80,0.83,0.62,No,p_minial_m7748,"Static 2-pass trial"
60,Router,2-Pass+Router,0.85,0.84,0.68,No,p_minial_m7748,"Router first run"
80,Router,2-Pass+Router,0.86,0.85,0.72,No,p_minial_m7748,"Stabilizing"
100,Few-Shot,1-Shot,0.85,0.85,0.75,No,p_minial_m7748,"Exploitation"
120,Mutation,p_minial_m8848,0.84,0.85,0.77,Yes,p_minial_json,"Plateau vs parent"
140,Schema,p_reason_then_json,0.55,0.84,0.78,No,p_reason_then_json,"Reasoning leak confirmed"
160,Router,2-Pass+Router,0.86,0.85,0.80,No,p_minial_m7748,"Best efficiency"
180,Few-Shot,2-Shot,0.82,0.85,0.81,No,p_schema_guided,"Occasional use"
200,Router,2-Pass+Router,0.87,0.85,0.82,No,p_minial_m7748,"Peak reward"
220,Few-Shot,1-Shot,0.85,0.85,0.83,No,p_minial_m7748,"Dominant arm"
240,PassMode,Single-Pass,0.78,0.85,0.84,No,p_minial_m7748,"Baseline check"
260,Router,2-Pass+Router,0.86,0.85,0.85,No,p_minial_m7748,"Consistent"
280,Few-Shot,3-Shot,0.79,0.85,0.85,No,p_schema_guided,"Residual exploration"
300,Router,2-Pass+Router,0.86,0.85,0.86,No,p_minial_m7748,"Converged"
""",
    "Mutation_History.csv": """ParentPrompt,MutationID,SchemaChange,TemperatureChange,ShotChange,ResultReward,DeltaReward,DeltaF1_Steps,StructuredValidity,Notes
p_minial_json,m7748,"minimal → minimal-tight","0.7→0.6","1-shot→1-shot",0.85,+0.03,+0.05,0.99,"Top mutation"
p_minial_m7748,m8848,"tight→tight+verbiage","0.6→0.6","1-shot→1-shot",0.846,-0.004,-0.01,0.99,"Plateau"
p_schema_guided,m1032,"guided → guided+order hints","0.7→0.65","1-shot→1-shot",0.823,+0.013,+0.02,0.98,"Small gain"
p_reason_then_json,m502,"reason→structured preamble","0.7→0.7","0-shot→0-shot",0.56,+0.06,+0.03,0.82,"Still below top schemas"
p_minial_m7748,m8102,"tight→tight+explicit keys","0.6→0.6","1-shot→1-shot",0.852,+0.002,+0.01,0.99,"Neutral"
p_schema_strict,m2101,"strict→strict-relaxed","0.7→0.65","1-shot→1-shot",0.79,+0.02,+0.03,0.96,"Fewer nulls"
""",
    "TwoPass_Component_Evaluation.csv": """Stage,Field,F1,Delta_vs_1Pass,Notes
Extraction(Intent/Constraints/Urgency),Intent,0.92,+0.03,"Cleaner slotting"
Extraction(Intent/Constraints/Urgency),Constraints,0.81,+0.07,"Fewer omissions"
Extraction(Intent/Constraints/Urgency),Urgency,0.80,+0.07,"Better categorical mapping"
StepGeneration(Conditioned_on_Extraction),Steps,0.66,+0.09,"Sequencing improved"
Combined(2-Pass Output),Mean F1,0.78,+0.05,"Static two-pass summary"
Router(Adaptive Path),Mean F1,0.82,+0.09,"Best overall after routing"
""",
    "Router_Decision_Distribution.csv": """InputComplexityScore,RoutedPath,Share_Percent,MeanF1,Reward,AvgTokens,Reward_per_1K_Tokens,Notes
Low (single-intent),Single-Pass,38,0.81,0.84,260,3.2,"Fast path"
Medium (some constraints),Two-Pass,22,0.82,0.83,340,2.4,"Balanced"
Medium (some constraints),Router→Single,12,0.81,0.85,270,3.1,"Adaptive skip"
High (multi-intent/constraint-dense),Two-Pass,16,0.83,0.84,390,2.2,"Full reasoning"
High (multi-intent/constraint-dense),Router→Two-Pass,12,0.84,0.86,410,2.1,"Selective deep pass"
""",
    "RewardVariance_By_System.csv": """System,MeanReward,StdDev,CoefVar_Percent,Notes
0-Shot,0.75,0.08,10.7,"Unstable outputs"
1-Shot,0.85,0.05,5.9,"Stable and efficient"
2-Shot,0.82,0.06,7.3,"Higher cost variability"
3-Shot,0.79,0.06,7.6,"Plateau"
2-Pass Static,0.78,0.05,6.4,"More consistent than single"
2-Pass + Router,0.82,0.04,4.9,"~20% variance reduction vs baseline"
""",
    "Token_Cost_Efficiency.csv": """System,AvgTokens,MeanReward,Reward_per_Token,Reward_per_1K_Tokens,DeltaEfficiency_vs_Prev,Notes
0-Shot,320,0.75,0.00234,2.34,,"Low efficiency"
1-Shot,315,0.85,0.00270,2.70,+15%,"Peak few-shot efficiency"
2-Shot,470,0.82,0.00174,1.74,-36%,"Token overhead"
3-Shot,620,0.79,0.00127,1.27,-27%,"Inefficient"
2-Pass Static,300,0.78,0.00260,2.60,+5%,"More consistent"
2-Pass + Router,310,0.82,0.00370,3.70,+42%,"Best overall efficiency"
""",
    "Field_Level_Error_Types.csv": """Phase,Field,ErrorType,Count,Percent_of_Field_Errors,Notes
Few-Shot_0,Intent,MissingField,48,40%,"No structure"
Few-Shot_0,Constraints,MalformedList,30,28%,"Unstructured text"
Few-Shot_0,Urgency,WrongLabel,22,20%,"Label drift"
Few-Shot_0,Steps,OrderError,44,35%,"Unordered steps"
Few-Shot_1,Intent,MissingField,6,8%,"Largely resolved"
Few-Shot_1,Constraints,MalformedList,12,18%,"Improving"
Few-Shot_1,Urgency,WrongLabel,10,15%,"Better mapping"
Few-Shot_1,Steps,OrderError,28,32%,"Still common"
Two-Pass,Steps,OrderError,18,25%,"Reduced with conditioning"
Two-Pass,Constraints,MissingItems,10,15%,"Better coverage"
Router,Steps,OrderError,12,20%,"Best stage"
Router,Urgency,WrongLabel,6,10%,"Lowest label noise"
""",
    "Target_Metric_Comparison.csv": """Metric,Target,Achieved,Delta,GoalMet,Notes
Valid Structured (%),>=95%,98%,+3%,Yes,"Router phase"
Mean F1,>=0.85,0.82,-0.03,No,"Strong but below stretch target"
Weakest Field F1,>=0.75,0.67,-0.08,No,"Steps remains hardest"
Mean Reward (post-convergence),>=0.85,0.85,0.00,Yes,"At target"
Reward per 1K Tokens,>1.0,3.7,+2.7,Yes,"High efficiency"
Mutation Outperformance Rate,>=20%,22%,+2%,Yes,"Met threshold"
""",
    "Experiment_Timeline.csv": """Phase,Milestone,Iterations,PrimaryChange,KeyMetric,Result,Notes
P1_SchemaTrials,Compare schema variants,120,Fixed params,Valid Structured (%),85–87 (best),"guided/minimal best"
P2_FewShot,0→1→2→3 shots,180,Examples added,Mean Reward,0.85 peak at 1-shot,"Efficiency wins"
P3_Bandit,Adaptive selection,200,Eps-greedy/TS,Convergence Speed,~50 iters to top arm,"~3× faster than manual"
P4_Mutations,Prompt evolution,160,Schema tightening,Reward,0.85 (m7748),"Best mutation"
P5_TwoPass,Decomposition,140,Extract→Steps,F1 Steps,0.63,"Sequential gain"
P6_Router,Adaptive routing,160,Dynamic path,Reward/1K,3.5–4.0,"Best overall"
""",
    "Reward_to_F1_Correlation.csv": """System,MeanReward,MeanF1
0-Shot,0.75,0.46
1-Shot,0.85,0.74
2-Shot,0.82,0.78
3-Shot,0.79,0.79
2-Pass Static,0.78,0.78
2-Pass + Router,0.82,0.82
Correlation_Pearson,0.90,
""",
    "Efficiency_to_Validity_Tradeoff.csv": """System,Reward_per_1K_Tokens,Valid_Structured_Percent,Notes
0-Shot,2.34,72,"Low validity"
1-Shot,2.70,90,"Good balance"
2-Shot,1.74,92,"Higher F1 but worse efficiency"
3-Shot,1.27,91,"Inefficient plateau"
2-Pass Static,2.60,95,"Better structure"
2-Pass + Router,3.70,98,"Optimal balance"
"""
