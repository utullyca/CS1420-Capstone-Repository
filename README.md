# Large Language Model Adaptation for Legal Writing Assessment

**A comparative study of few-shot prompting and fine-tuning approaches using Mistral 7B Instruct for automated legal writing feedback generation**

## Project Overview

This capstone project addresses a critical challenge in legal education: providing scalable, individualized feedback to first-year law students struggling with legal writing. Students entering law school with underdeveloped writing skills face significant barriers to success, often leading to scholarship loss or academic dismissal—outcomes that disproportionately affect students from marginalized backgrounds.

Legal Research & Writing (LRW) professors are a limited resource, typically teaching classes of 15-25 students. This constraint creates a bottleneck where:

- Students with weaker writing skills must use precious office hours for basic remediation
- Advanced writers receive less guidance on sophisticated legal analysis
- All students would benefit from unlimited access to constructive feedback

This project evaluates the effectiveness of Mistral 7B Instruct in generating automated feedback for legal memoranda through:

1. **Comparative analysis** of few-shot prompting vs. few-shot fine-tuning approaches
2. **Synthetic data generation** using a modified SoftSRV framework to expand limited training datasets
3. **Section-specific optimization** for different IREAC components (Issue, Rule, Explanation, Application, Conclusion)

### Key Findings

- **Empirical evidence** that fine-tuned models excel at structured evaluation tasks (100% accuracy on issue statements)
- **Novel insights** showing few-shot prompting performs better on complex reasoning tasks like rule application
- **Critical evaluation** of synthetic data generation, revealing 80% hallucination rates in legal contexts
- **Practical framework** for implementing hybrid approaches in legal education settings

## Research Questions

1. Which approach (few-shot prompting vs. fine-tuning) produces the most accurate and useful feedback on legal memorandum assignments?
2. Are synthetic legal texts generated via the SoftSRV framework viable as training or evaluation data?
