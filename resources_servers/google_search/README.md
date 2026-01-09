# Google Search Resource Server

## Table of Contents
- [Google Search Resource Server](#google-search-resource-server)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
    - [Overview](#overview)
    - [Dataset Description](#dataset-description)
    - [Environment Tools](#environment-tools)
    - [Verifier](#verifier)
    - [Legal Status](#legal-status)
  - [Requirements](#requirements)
  - [Environment Inheritance](#environment-inheritance)
  - [Usage Commands](#usage-commands)
    - [Running the Server](#running-the-server)
  - [Samples](#samples)
    - [Sample Prompts](#sample-prompts)
  - [Complete Rollout](#complete-rollout)
  - [Prompts (truncated) + Rewards](#prompts-truncated--rewards)
  - [Licensing Information](#licensing-information)
    - [Licensing information](#licensing-information-1)

## Description

### Overview
Reference documentation: [Link1](https://docs.google.com/spreadsheets/d/1VK4-ZonMSR-4Ulk161Au1f-nGhs4r9V9-beScboaR3I/edit?gid=235748913#gid=235748913&range=A7) | [Link2](https://docs.google.com/spreadsheets/d/1VK4-ZonMSR-4Ulk161Au1f-nGhs4r9V9-beScboaR3I/edit?gid=235748913#gid=235748913&range=A8)

### Dataset Description

You can find the example datast for this environment here: https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa 

The dataset has been filtering using Qwen3-32B for difficulty, ensuring that samples are not easily answerable without external search assistance. The datasets consist of Multiple Choice Question Answering (MCQA) tasks in the STEM domain. 

### Environment Tools
The environment provides two primary tools for information retrieval:

- **search**: Powered by Google's Programmable Search Engine ([documentation](https://developers.google.com/custom-search/docs/overview)). This API wrapper enables internet searches and returns structured results containing webpage summaries and corresponding links.
- **browse**: Utilizes trafilatura to extract and return cleaned content from webpages.

These tools enable models to perform comprehensive internet research by searching for relevant information and accessing detailed webpage content.

### Verifier
The verifier checks if the boxed MCQ option is equal to the ground truth option.

### Legal Status
**Approved**: Yes

## Requirements
This environment uses [Programmable Search Engine](https://developers.google.com/custom-search/v1/using_rest). You will require a `google_api_key` and `google_cx` to use this environment, which you must add in your `env.yaml`. 
```
google_search:
  resources_servers:
    google_search:
      google_api_key: <your_api_key>
      google_cx: <your_cx_engine>
```
Please reach out to @rgala or @fsoares if you require access credentials. 

## Environment Inheritance
This environment can be inherited to integrate search functionality into your own custom environment. The following tool definitions are available for inheritance:
```
tools=[
    {
        "type": "function",
        "name": "search",
        "description": "Search Google for a query and return up to 10 search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to search for",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "browse",
        "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The url of the page to get the content of"}
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]
```

## Usage Commands

### Running the Server
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=data/Nemotron-RL-knowledge-web_search-mcqa.jsonl \
    +output_jsonl_fpath=results/Nemotron-RL-knowledge-web_search-mcqa.jsonl \
    +limit=1

ng_viewer +jsonl_fpath=results/Nemotron-RL-knowledge-web_search-mcqa.jsonl

```

## Samples

### Sample Prompts
```
{"responses_create_params": {"instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.", "input": "In cardiac transplantation, a patient presents with progressive Chronic Allograft Vasculopathy (CAV) confirmed by serial IVUS showing accelerated intimal thickening. The patient has stable immunosuppression levels but a history of CMV infection and poorly controlled dyslipidemia. Which factor is the strongest independent predictor of irreversible CAV progression in this scenario?\nA: Persistent donor-specific HLA antibodies despite plasmapheresis\nB: Intimal hyperplasia >0.5 mm on year 1 surveillance angiography\nC: Concomitant de novo Class II DSA development\nD: Elevated C-reactive protein >3 mg/L on serial measurements\nE: Cytomegalovirus (CMV) viremia \u226510,000 copies/mL\nF: Media layer inflammation with macrophage infiltration\nG: Dyslipidemia with LDL >130 mg/dL despite statin therapy\nH: Downregulation of endothelial protective genes (FOXP3, HMOX1)\nI: Adventitial neovascularization >10 microvessels/hpf\nJ: Recipient CCR5\u039432 homozygous mutation", "tools": [{"type": "function", "name": "search", "description": "Search Google for a query and return up to 10 search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The term to search for"}}, "required": ["query"], "additionalProperties": false}, "strict": true}, {"type": "function", "name": "browse", "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}}, "required": ["url"], "additionalProperties": false}, "strict": true}], "max_output_tokens": 32768, "temperature": 0.6, "parallel_tool_calls": false}, "expected_answer": "C", "task_difficulty_qwen3_32b_avg_8": 0.5}
{"responses_create_params": {"instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.", "input": "Consider a quantum system with a non-degenerate ground state undergoing adiabatic cyclic evolution governed by parameters \\(\\mathbf{R}(t)\\). Let \\(\\gamma_n\\) be the Berry phase for the n-th eigenstate and \\(\\mathbf{A}_n(\\mathbf{R}) = i\\langle n(\\mathbf{R})|\\nabla_\\mathbf{R} n(\\mathbf{R})\\rangle\\) be the Berry connection. Which statement is correct regarding gauge invariance and properties of the Berry phase?\nA: The Berry phase \\(\\gamma_n\\) depends explicitly on the chosen gauge\nB: The Berry connection \\(\\mathbf{A}_n(\\mathbf{R})\\) is gauge invariant\nC: The Berry phase vanishes identically for non-cyclic evolutions\nD: The Berry phase \\(\\gamma_n\\) includes contributions from both geometric and dynamic phases\nE: For Abelian Berry phases, \\(\\nabla_\\mathbf{R} \\times \\mathbf{A}_n(\\mathbf{R}) = 0\\) always holds\nF: The Berry curvature \\(\\mathbf{B}_n(\\mathbf{R})\\) is gauge-dependent\nG: The Berry phase \\(\\gamma_n\\) is gauge invariant modulo \\(2\\pi\\)\nH: The Berry phase accumulates only when the Hamiltonian returns to itself at \\(t=T\\)\nI: For degenerate states, the Berry phase reduces to a U(1) phase factor\nJ: The non-Abelian Berry phase is always commutative under gauge transformations", "tools": [{"type": "function", "name": "search", "description": "Search Google for a query and return up to 10 search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The term to search for"}}, "required": ["query"], "additionalProperties": false}, "strict": true}, {"type": "function", "name": "browse", "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}}, "required": ["url"], "additionalProperties": false}, "strict": true}], "max_output_tokens": 32768, "temperature": 0.6, "parallel_tool_calls": false}, "expected_answer": "G", "task_difficulty_qwen3_32b_avg_8": 0.375}
{"responses_create_params": {"instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.", "input": "When utilizing Virtual Surgical Planning (VSP) for complex mandibular reconstruction with a fibula free flap following segmental resection and simultaneous orthognathic surgery (bimaxillary osteotomy), which of the following represents the **greatest** inherent limitation in achieving precise transfer of the virtual plan to the operative field, assuming ideal technical execution and equipment function?\nA: Inaccuracies in preoperative CT/CBCT segmentation leading to erroneous defect volume calculation\nB: The inability of cutting guides to accommodate natural curvature variations in the harvested fibula segment\nC: Error propagation from superimposing preoperative images onto intraoperative patient anatomy via surface registration\nD: Inherent limitation of stereolithographic guides in controlling the rotational position of fibula segments relative to the occlusal plane\nE: Unpredictable soft tissue influence on final skeletal position post-inset and fixation after mandibular repositioning during orthognathic correction\nF: Cumulative vectorial error compounding during the sequence: virtual positioning of fibula segments \u2192 physical cutting guide fabrication \u2192 mandibular repositioning post-Le Fort I \u2192 fibula flap inset\nG: Difficulty in achieving precise adaptation between fibula osteotomy surfaces and recipient mandibular resection margins due to bone density mismatch\nH: Standard software algorithms' inability to account for intraoperative muscle traction forces altering bone segment position post-osteotomy\nI: Inconsistent fidelity of registration markers due to anatomical deformation after tumor resection prior to flap inset\nJ: Limitations of intraoperative navigation systems in tracking multiple independently moving segments (maxilla, proximal/distal mandible, fibula segments) simultaneously", "tools": [{"type": "function", "name": "search", "description": "Search Google for a query and return up to 10 search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The term to search for"}}, "required": ["query"], "additionalProperties": false}, "strict": true}, {"type": "function", "name": "browse", "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}}, "required": ["url"], "additionalProperties": false}, "strict": true}], "max_output_tokens": 32768, "temperature": 0.6, "parallel_tool_calls": false}, "expected_answer": "F", "task_difficulty_qwen3_32b_avg_8": 0.375}
{"responses_create_params": {"instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.", "input": "In a migratory passerine species studied over 10 years, Lack's principle predicts clutch size evolution maximizes the number of fledglings surviving to recruitment (returning to breed). Fledgling recruitment probability (S) exhibits nonlinear density dependence: S = 0.7 / (1 + k*N), where N is the number of fledglings produced locally and k is a scaling constant (k=0.005 per fledgling). However, empirical data reveals an additional carry-over effect: parental investment per egg (I) = 1000 - 10*(C-4) kJ (for C>=4), where C is clutch size. Eggs receiving I < 970 kJ produce fledglings with a 30% *reduction* in their base survival probability *during the subsequent non-breeding season migration* (applied independently of local fledgling density). This reduction multiplies the base recruitment probability S. Recruitment is further subject to stochastic variation (\u00b10.05) around the calculated S for each offspring. Accounting simultaneously for immediate density-dependent survival and the threshold-based carry-over effect influencing future survival, which clutch size C (integer) results in the *highest geometric mean recruitment success per breeding attempt* (fledglings surviving to recruit) over the long term under this model? Assume constant adult survival and no nest predation variation.\nA: 5 eggs\nB: 6 eggs\nC: 7 eggs\nD: 8 eggs", "tools": [{"type": "function", "name": "search", "description": "Search Google for a query and return up to 10 search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The term to search for"}}, "required": ["query"], "additionalProperties": false}, "strict": true}, {"type": "function", "name": "browse", "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}}, "required": ["url"], "additionalProperties": false}, "strict": true}], "max_output_tokens": 32768, "temperature": 0.6, "parallel_tool_calls": false}, "expected_answer": "C", "task_difficulty_qwen3_32b_avg_8": 0.25}
{"responses_create_params": {"instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.", "input": "Consider the one-dimensional Fokker-Planck equation for the probability density \\( P(x,t) \\):\n\\[\n\\frac{\\partial P}{\\partial t} = - \\frac{\\partial}{\\partial x} \\left[ \\mu(x) P \\right] + \\frac{\\partial^2}{\\partial x^2} \\left[ D(x) P \\right]\n\\]\nwhere \\( \\mu(x) = -U'(x) \\) for a potential \\( U(x) \\), and \\( D(x) > 0 \\) is the diffusion coefficient. Assume that at steady state, \\( P_{\\text{st}}(x) \\) exists and satisfies the reflecting boundary condition \\( J_{\\text{st}}(a) = 0 \\) at \\( x = a \\), where \\( J_{\\text{st}}(x) \\) is the probability current. Which of the following expressions represents the stationary probability current \\( J_{\\text{st}}(x) \\) at any position \\( x \\)?\nA: \\( J_{\\text{st}}(x) = 0 \\)\nB: \\( J_{\\text{st}}(x) = -\\mu(x) P_{\\text{st}}(x) - \\frac{d}{dx}\\left[D(x) P_{\\text{st}}(x)\\right] \\)\nC: \\( J_{\\text{st}}(x) = \\mu(x) P_{\\text{st}}(x) - D(x) \\frac{dP_{\\text{st}}}{dx} \\)\nD: \\( J_{\\text{st}}(x) = -\\mu(x) P_{\\text{st}}(x) + D(x) \\frac{dP_{\\text{st}}}{dx} \\)\nE: \\( J_{\\text{st}}(x) = -\\mu(x) P_{\\text{st}}(x) - D(x) \\frac{dP_{\\text{st}}}{dx} \\)\nF: \\( J_{\\text{st}}(x) = -\\frac{dU}{dx} \\)\nG: \\( J_{\\text{st}}(x) = \\frac{d}{dx}\\left[ D(x) P_{\\text{st}}(x) \\right] \\)\nH: \\( J_{\\text{st}}(x) = -D(x) \\frac{dP_{\\text{st}}}{dx} \\)\nI: \\( J_{\\text{st}}(x) = k \\) (some constant k)\nJ: None of the above", "tools": [{"type": "function", "name": "search", "description": "Search Google for a query and return up to 10 search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The term to search for"}}, "required": ["query"], "additionalProperties": false}, "strict": true}, {"type": "function", "name": "browse", "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}}, "required": ["url"], "additionalProperties": false}, "strict": true}], "max_output_tokens": 32768, "temperature": 0.6, "parallel_tool_calls": false}, "expected_answer": "A", "task_difficulty_qwen3_32b_avg_8": 0.25}
```


## Complete Rollout
```
{
  "responses_create_params": {
    "input": [
      {
        "content": "When utilizing Virtual Surgical Planning (VSP) for complex mandibular reconstruction with a fibula free flap following segmental resection and simultaneous orthognathic surgery (bimaxillary osteotomy), which of the following represents the **greatest** inherent limitation in achieving precise transfer of the virtual plan to the operative field, assuming ideal technical execution and equipment function?\n\nA: Inaccuracies in preoperative CT/CBCT segmentation leading to erroneous defect volume calculation\nB: The inability of cutting guides to accommodate natural curvature variations in the harvested fibula segment\nC: Error propagation from superimposing preoperative images onto intraoperative patient anatomy via surface registration\nD: Inherent limitation of stereolithographic guides in controlling the rotational position of fibula segments relative to the occlusal plane\nE: Unpredictable soft tissue influence on final skeletal position post-inset and fixation after mandibular repositioning during orthognathic correction\nF: Cumulative vectorial error compounding during the sequence: virtual positioning of fibula segments → physical cutting guide fabrication → mandibular repositioning post-Le Fort I → fibula flap inset\nG: Difficulty in achieving precise adaptation between fibula osteotomy surfaces and recipient mandibular resection margins due to bone density mismatch\nH: Standard software algorithms' inability to account for intraoperative muscle traction forces altering bone segment position post-osteotomy\nI: Inconsistent fidelity of registration markers due to anatomical deformation after tumor resection prior to flap inset\nJ: Limitations of intraoperative navigation systems in tracking multiple independently moving segments (maxilla, proximal/distal mandible, fibula segments) simultaneously",
        "role": "user",
        "type": "message"
      }
    ],
    "instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.",
    "max_output_tokens": 32768,
    "parallel_tool_calls": false,
    "temperature": 0.6,
    "tools": [
      {
        "name": "search",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The term to search for"
            }
          },
          "required": ["query"],
          "additionalProperties": false
        },
        "strict": true,
        "type": "function",
        "description": "Search Google for a query and return up to 10 search results."
      },
      {
        "name": "browse",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "The url of the page to get the content of"
            }
          },
          "required": ["url"],
          "additionalProperties": false
        },
        "strict": true,
        "type": "function",
        "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words."
      }
    ]
  },
  "response": {
    "id": "resp_68a4e16f217c819789717b01c5be955806396735271e9c27",
    "created_at": 1755636079.0,
    "error": null,
    "incomplete_details": null,
    "instructions": "Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from google. You can then use the result(s) you find helpful to either `browse` or `search` again. You are encouraged to use tools to have the most grounded answer.",
    "metadata": {},
    "model": "gpt-4.1-2025-04-14",
    "object": "response",
    "output": [
      {
        "arguments": "{\"query\":\"greatest limitation of VSP in mandibular reconstruction with fibula free flap and simultaneous orthognathic surgery\"}",
        "call_id": "call_zoVxSzVfmlhWWt46az6osYhJ",
        "name": "search",
        "type": "function_call",
        "id": "fc_68a4e163944881978e00afcc0dc2ad1706396735271e9c27",
        "status": "completed"
      },
      {
        "call_id": "call_zoVxSzVfmlhWWt46az6osYhJ",
        "output": "{\"search_results\": \"{\\\"kind\\\": \\\"customsearch#search\\\", \\\"url\\\": {\\\"type\\\": \\\"application/json\\\", \\\"template\\\": \\\"https://www.googleapis.com/customsearch/v1?q={searchTerms}&num={count?}&start={startIndex?}...\\\"}, \\\"queries\\\": {\\\"request\\\": [{\\\"title\\\": \\\"Google Custom Search - greatest limitation of VSP in mandibular reconstruction with fibula free flap and simultaneous orthognathic surgery\\\", \\\"totalResults\\\": \\\"3750\\\", \\\"searchTerms\\\": \\\"greatest limitation of VSP in mandibular reconstruction with fibula free flap and simultaneous orthognathic surgery\\\", \\\"count\\\": 10, \\\"startIndex\\\": 1, \\\"inputEncoding\\\": \\\"utf8\\\", \\\"outputEncoding\\\": \\\"utf8\\\", \\\"safe\\\": \\\"off\\\", \\\"cx\\\": \\\"c7287d4deda604fd6\\\"}]}, \\\"context\\\": {\\\"title\\\": \\\"post-training-web\\\"}, \\\"searchInformation\\\": {\\\"searchTime\\\": 0.239275, \\\"formattedSearchTime\\\": \\\"0.24\\\", \\\"totalResults\\\": \\\"3750\\\", \\\"formattedTotalResults\\\": \\\"3,750\\\"}, \\\"items\\\": [{\\\"kind\\\": \\\"customsearch#result\\\", \\\"title\\\": \\\"Mandibular Reconstruction With Fibula Flap and Dental Implants ...\\\", \\\"htmlTitle\\\": \\\"<b>Mandibular Reconstruction With Fibula Flap</b> and Dental Implants ...\\\", \\\"link\\\": \\\"https://pmc.ncbi.nlm.nih.gov/articles/PMC8525397/\\\", \\\"displayLink\\\": \\\"pmc.ncbi.nlm.nih.gov\\\", \\\"snippet\\\": \\\"Oct 5, 2021 ... The major limitation of VSP is the added cost of the surgical procedure ... Mandibular Reconstruction With Free Fibula Flap. J Oral\\\\u00a0...\\\", \\\"htmlSnippet\\\": \\\"Oct 5, 2021 <b>...</b> The major <b>limitation of VSP</b> is the added cost of the <b>surgical procedure</b> ... <b>Mandibular Reconstruction</b> With <b>Free Fibula Flap</b>. J <b>Oral</b>&nbsp;...\\\", \\\"formattedUrl\\\": \\\"https://pmc.ncbi.nlm.nih.gov/articles/PMC8525397/\\\", \\\"htmlFormattedUrl\\\": \\\"https://pmc.ncbi.nlm.nih.gov/articles/PMC8525397/\\\"}, {\\\"kind\\\": \\\"customsearch#result\\\", \\\"title\\\": \\\"Virtual surgical planning in fibula flap mandibular reconstruction ...\\\", \\\"htmlTitle\\\": \\\"Virtual <b>surgical</b> planning in <b>fibula flap mandibular reconstruction</b> ...\\\", \\\"link\\\": \\\"https://fomm.amegroups.org/article/view/42414/html\\\", \\\"displayLink\\\": \\\"fomm.amegroups.org\\\", \\\"snippet\\\": \\\"Mandibular reconstruction is one of the workhorse in head and neck reconstructive surgery, with the free fibula flap currently considered the gold standard\\\\u00a0...\\\", \\\"htmlSnippet\\\": \\\"<b>Mandibular reconstruction</b> is one of the workhorse in head and neck reconstructive <b>surgery</b>, with the <b>free fibula flap</b> currently considered the gold standard&nbsp;...\\\", \\\"formattedUrl\\\": \\\"https://fomm.amegroups.org/article/view/42414/html\\\", \\\"htmlFormattedUrl\\\": \\\"https://fomm.amegroups.org/article/view/42414/html\\\"}",
        "type": "function_call_output",
        "id": null,
        "status": null
      },
      {
        "arguments": "{\"url\":\"https://fomm.amegroups.org/article/view/42414/html\"}",
        "call_id": "call_wRdfS2b5zN1lM8eL7DyLK7qu",
        "name": "browse",
        "type": "function_call",
        "id": "fc_68a4e167ce588197935f3aa71c7a485606396735271e9c27",
        "status": "completed"
      },
      {
        "call_id": "call_wRdfS2b5zN1lM8eL7DyLK7qu",
        "output": "[TRUNCATED - Full article content about VSP in fibula flap reconstruction, discussing methodology, advantages, disadvantages, and accuracy considerations. Key points include: cleaner surgical margins, decreased operative time, improved aesthetic outcomes, but increased costs and no improvement in complication rates.]",
        "type": "function_call_output",
        "id": null,
        "status": null
      },
      {
        "arguments": "{\"url\":\"https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.743389/full\"}",
        "call_id": "call_IEt0r6nES20fTAaed0QsIWGo",
        "name": "browse",
        "type": "function_call",
        "id": "fc_68a4e16ba4f08197966584ad6b7923b906396735271e9c27",
        "status": "completed"
      },
      {
        "call_id": "call_IEt0r6nES20fTAaed0QsIWGo",
        "output": "[TRUNCATED - Research article about novel fibula malleolus cap for improving accuracy in oncologic jaw reconstruction. Study shows significant reduction in sliding and rotational errors, improving implant placement accuracy from 3.2mm to 1.3mm platform deviation and angle errors from 11.3° to 4.6°.]",
        "type": "function_call_output",
        "id": null,
        "status": null
      },
      {
        "id": "msg_68a4e17063ac81979270cd9e0ac5c0a106396735271e9c27",
        "content": [
          {
            "annotations": [],
            "text": "Based on the literature and detailed review of the limitations in virtual surgical planning (VSP) for mandibular reconstruction with a fibula free flap and simultaneous orthognathic surgery, the **greatest inherent limitation** (assuming ideal technical execution and equipment function) is the **unpredictable influence of soft tissue on final skeletal position** after the inset and fixation, particularly when both mandibular reconstruction and bimaxillary osteotomy are performed together.\n\nKey points from the literature:\n- Most technical and hardware-based errors (like CT segmentation, registration, guide fit, bone mismatch, or hardware tracking) can be minimized with precise execution and improved technology.\n- Even with perfect bony alignment and fixation, **soft tissue tension, scarring, and intraoperative manipulation can unpredictably alter the final position of the reconstructed mandible and maxilla**, especially after simultaneous orthognathic movements.\n- This unpredictable soft tissue behavior is not currently modeled or correctable by VSP or intraoperative navigation, making it the biggest \"uncontrollable\" variable in transferring the virtual plan to the actual patient outcome.\n\nThis matches option E:\n\n**E: Unpredictable soft tissue influence on final skeletal position post-inset and fixation after mandibular repositioning during orthognathic correction**\n\nTherefore, the answer is\n\n\\boxed{E}",
            "type": "output_text",
            "logprobs": []
          }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
      }
    ],
    "parallel_tool_calls": false,
    "temperature": 0.6,
    "tool_choice": "auto",
    "tools": [
      {
        "name": "search",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The term to search for"
            }
          },
          "required": ["query"],
          "additionalProperties": false
        },
        "strict": true,
        "type": "function",
        "description": "Search Google for a query and return up to 10 search results."
      },
      {
        "name": "browse",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "The url of the page to get the content of"
            }
          },
          "required": ["url"],
          "additionalProperties": false
        },
        "strict": true,
        "type": "function",
        "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words."
      }
    ],
    "top_p": 1.0,
    "background": false,
    "max_output_tokens": 32768,
    "max_tool_calls": null,
    "previous_response_id": null,
    "prompt": null,
    "reasoning": {
      "effort": null,
      "generate_summary": null,
      "summary": null
    },
    "service_tier": "default",
    "status": "completed",
    "text": {
      "format": {
        "type": "text"
      },
      "verbosity": "medium"
    },
    "top_logprobs": 0,
    "truncation": "disabled",
    "usage": {
      "input_tokens": 24751,
      "input_tokens_details": {
        "cached_tokens": 15616
      },
      "output_tokens": 266,
      "output_tokens_details": {
        "reasoning_tokens": 0
      },
      "total_tokens": 25017
    },
    "user": null,
    "prompt_cache_key": null,
    "safety_identifier": null,
    "store": true
  },
  "reward": 0.0,
  "parsed_option": "E"
}
```

## Prompts (truncated) + Rewards
```
{
    "create_params_question_only": "In cardiac transplantation, a patient presents with progressive Chronic Allograft Vasculopathy (CAV) confirmed by serial IVUS showing accelerated intimal thickening. The patient has stable immunosuppression levels but a history of CMV infection and poorly controlled dyslipidemia. Which factor is the strongest independent predictor of irreversible CAV progression in this scenario?",
    "expected_answer": "C",
    "actual_response": "The strongest independent predictor of irreversible CAV progression in the scenario described is:\n\n**B: Intimal hyperplasia >0.5 mm on year 1 surveillance angiography**\n\nGrounding:\n- Multiple high-quality sources, including a 2005 multicenter IVUS study and review articles, confirm that an increase in intimal thickness of \u22650.5 mm within the first year after transplant (measured by IVUS) is a powerful predictor of future CAV progression, graft loss, and mortality. This marker is considered the best independent predictor of adverse outcomes related to CAV.\n- For example, see: [JACC 2005; 45:1532\u20131537](https://www.sciencedirect.com/science/article/pii/S0735109705004651), [PubMed ID: 15862431](https://pubmed.ncbi.nlm.nih.gov/15862431/).\n\nOther options, such as donor-specific HLA antibodies or CMV viremia, are important risk factors but not as strong or as independently predictive as early IVUS-detected intimal thickening.\n\nFinal answer:  \\(\\boxed{B}\\)",
    "parsed_option": "B",
    "reward": 0.0,
    "difficulty": 0.5
}
{
    "create_params_question_only": "Consider a quantum system with a non-degenerate ground state undergoing adiabatic cyclic evolution governed by parameters \\(\\mathbf{R}(t)\\). Let \\(\\gamma_n\\) be the Berry phase for the n-th eigenstate and \\(\\mathbf{A}_n(\\mathbf{R}) = i\\langle n(\\mathbf{R})|\\nabla_\\mathbf{R} n(\\mathbf{R})\\rangle\\) be the Berry connection. Which statement is correct regarding gauge invariance and properties of the Berry phase?",
    "expected_answer": "G",
    "actual_response": "Based on multiple authoritative sources, the Berry phase \\(\\gamma_n\\) is gauge invariant modulo \\(2\\pi\\). This is a fundamental property of the Berry phase in quantum mechanics, as discussed in standard references and review articles on the subject.\n\nTherefore, the correct answer is:\n\\[\n\\boxed{G}\n\\]",
    "parsed_option": "G",
    "reward": 1.0,
    "difficulty": 0.375
}
{
    "create_params_question_only": "When utilizing Virtual Surgical Planning (VSP) for complex mandibular reconstruction with a fibula free flap following segmental resection and simultaneous orthognathic surgery (bimaxillary osteotomy), which of the following represents the **greatest** inherent limitation in achieving precise transfer of the virtual plan to the operative field, assuming ideal technical execution and equipment function?",
    "expected_answer": "F",
    "actual_response": "Based on the literature and detailed review of the limitations in virtual surgical planning (VSP) for mandibular reconstruction with a fibula free flap and simultaneous orthognathic surgery, the **greatest inherent limitation** (assuming ideal technical execution and equipment function) is the **unpredictable influence of soft tissue on final skeletal position** after the inset and fixation, particularly when both mandibular reconstruction and bimaxillary osteotomy are performed together.\n\nKey points from the literature:\n- Most technical and hardware-based errors (like CT segmentation, registration, guide fit, bone mismatch, or hardware tracking) can be minimized with precise execution and improved technology.\n- Even with perfect bony alignment and fixation, **soft tissue tension, scarring, and intraoperative manipulation can unpredictably alter the final position of the reconstructed mandible and maxilla**, especially after simultaneous orthognathic movements.\n- This unpredictable soft tissue behavior is not currently modeled or correctable by VSP or intraoperative navigation, making it the biggest \"uncontrollable\" variable in transferring the virtual plan to the actual patient outcome.\n\nThis matches option E:\n\n**E: Unpredictable soft tissue influence on final skeletal position post-inset and fixation after mandibular repositioning during orthognathic correction**\n\nTherefore, the answer is\n\n\\boxed{E}",
    "parsed_option": "E",
    "reward": 0.0,
    "difficulty": 0.375
}
{
    "create_params_question_only": "In a migratory passerine species studied over 10 years, Lack's principle predicts clutch size evolution maximizes the number of fledglings surviving to recruitment (returning to breed). Fledgling recruitment probability (S) exhibits nonlinear density dependence: S = 0.7 / (1 + k*N), where N is the number of fledglings produced locally and k is a scaling constant (k=0.005 per fledgling). However, empirical data reveals an additional carry-over effect: parental investment per egg (I) = 1000 - 10*(",
    "expected_answer": "C",
    "actual_response": "Let's solve the problem step by step:\n\n### 1. **Density-Dependent Recruitment Probability (S)**\nGiven:\n- \\( S = \\frac{0.7}{1 + kN} \\)\n- \\( k = 0.005 \\) per fledgling\n- \\( N = \\) number of fledglings produced locally (equals clutch size \\( C \\) if all survive to fledging)\n\n### 2. **Parental Investment per Egg (I)**\n- \\( I = 1000 - 10 \\times (C-4) \\) kJ, for \\( C \\ge 4 \\)\n\n### 3. **Carry-over Effect**\n- If \\( I < 970 \\) kJ, then survival probability is reduced by 30% during migration (i.e., multiply S by 0.7).\n- \\( I < 970 \\) kJ when \\( 1000 - 10(C-4) < 970 \\implies 10(C-4) > 30 \\implies C > 7 \\).\n- So, for \\( C \\leq 7 \\), **no carry-over penalty**; for \\( C \\geq 8 \\), **carry-over penalty applies**.\n\n### 4. **Stochastic Variation**\n- \u00b10.05 around S per offspring, but for *geometric mean* over many years, we focus on the mean S (as variation averages out).\n\n### 5. **Recruitment per Breeding Attempt**\n- Recruitment per attempt = \\( C \\times S \\) (if no penalty), or \\( C \\times S \\times 0.7 \\) (if penalty applies).\n\n---\n\n## **Compute for Each Clutch Size (C = 5, 6, 7, 8)**\n\n### **A. C = 5**\n- \\( N = 5 \\)\n- \\( S = \\frac{0.7}{1 + 0.005 \\times 5} = \\frac{0.7}{1.025} \\approx 0.6829 \\)\n- \\( I = 1000 - 10(5-4) = 990 \\) kJ (**no penalty, since 990 > 970**)\n- Recruitment \\( = 5 \\times 0.6829 = 3.414 \\)\n\n### **B. C = 6**\n- \\( N = 6 \\)\n- \\( S = \\frac{0.7}{1 + 0.005 \\times 6} = \\frac{0.7}{1.03} \\approx 0.6796 \\)\n- \\( I = 1000 - 10(6-4) = 980 \\) kJ (**no penalty, since 980 > 970**)\n- Recruitment \\( = 6 \\times 0.6796 = 4.078 \\)\n\n### **C. C = 7**\n- \\( N = 7 \\)\n- \\( S = \\frac{0.7}{1 + 0.005 \\times 7} = \\frac{0.7}{1.035} \\approx 0.6763 \\)\n- \\( I = 1000 - 10(7-4) = 970 \\) kJ (**no penalty, since 970 = 970**)\n- Recruitment \\( = 7 \\times 0.6763 = 4.734 \\)\n\n### **D. C = 8**\n- \\( N = 8 \\)\n- \\( S = \\frac{0.7}{1 + 0.005 \\times 8} = \\frac{0.7}{1.04} \\approx 0.6731 \\)\n- \\( I = 1000 - 10(8-4) = 960 \\) kJ (**penalty applies, since 960 < 970**)\n- Adjusted S \\( = 0.6731 \\times 0.7 \\approx 0.4712 \\)\n- Recruitment \\( = 8 \\times 0.4712 = 3.770 \\)\n\n---\n\n## **Summary Table**\n\n| C | S (before penalty) | I (kJ) | Penalty | S (after penalty) | Recruitment |\n|---|--------------------|--------|---------|-------------------|-------------|\n| 5 | 0.6829             | 990    | No      | 0.6829            | 3.414       |\n| 6 | 0.6796             | 980    | No      | 0.6796            | 4.078       |\n| 7 | 0.6763             | 970    | No      | 0.6763            | 4.734       |\n| 8 | 0.6731             | 960    | Yes     | 0.4712            | 3.770       |\n\n---\n\n## **Conclusion**\n\n- **C = 7** eggs gives the highest recruitment per breeding attempt, as the penalty for C = 8 reduces its value below that of C = 7.\n\nThe answer is:\n\n\\[\n\\boxed{C}\n\\]",
    "parsed_option": "C",
    "reward": 1.0,
    "difficulty": 0.25
}
{
    "create_params_question_only": "Consider the one-dimensional Fokker-Planck equation for the probability density \\( P(x,t) \\):",
    "expected_answer": "A",
    "actual_response": "I was not able to retrieve the PDF contents directly, but based on standard results for the Fokker-Planck equation with position-dependent diffusion, the probability current is typically given by:\n\n\\[\nJ_{\\text{st}}(x) = \\mu(x) P_{\\text{st}}(x) - \\frac{d}{dx}\\left[D(x) P_{\\text{st}}(x)\\right]\n\\]\n\nBut let's clarify with the standard form. The Fokker-Planck equation is:\n\n\\[\n\\frac{\\partial P}{\\partial t} = -\\frac{\\partial}{\\partial x}[\\mu(x)P] + \\frac{\\partial^2}{\\partial x^2}[D(x)P]\n\\]\n\nThe probability current is defined such that:\n\n\\[\n\\frac{\\partial P}{\\partial t} = -\\frac{\\partial J}{\\partial x}\n\\]\n\nSo,\n\n\\[\nJ(x, t) = \\mu(x)P(x, t) - \\frac{\\partial}{\\partial x}[D(x)P(x, t)]\n\\]\n\nAt steady state (\\( P_{\\text{st}} \\)), the same form holds:\n\n\\[\nJ_{\\text{st}}(x) = \\mu(x)P_{\\text{st}}(x) - \\frac{d}{dx}[D(x)P_{\\text{st}}(x)]\n\\]\n\nComparing to the answer choices:\n- Option C: \\( J_{\\text{st}}(x) = \\mu(x) P_{\\text{st}}(x) - D(x) \\frac{dP_{\\text{st}}}{dx} \\)\n- Option D: \\( J_{\\text{st}}(x) = -\\mu(x) P_{\\text{st}}(x) + D(x) \\frac{dP_{\\text{st}}}{dx} \\)\n- Option B: \\( J_{\\text{st}}(x) = -\\mu(x) P_{\\text{st}}(x) - \\frac{d}{dx}\\left[D(x) P_{\\text{st}}(x)\\right] \\)\n\nBut the correct standard form is:\n\n\\[\nJ_{\\text{st}}(x) = \\mu(x) P_{\\text{st}}(x) - \\frac{d}{dx}[D(x) P_{\\text{st}}(x)]\n\\]\n\nThis matches exactly with **option C** if and only if \\( D(x) \\) is constant, otherwise it is not exactly the same. Option C does not have the full derivative, so let's check carefully.\n\nExpanding option C:\n\\[\nJ(x) = \\mu(x)P(x) - D(x)\\frac{dP}{dx}\n\\]\n\nExpanding the correct form:\n\\[\nJ(x) = \\mu(x)P(x) - \\frac{d}{dx}[D(x)P(x)] = \\mu(x)P(x) - D(x)\\frac{dP}{dx} - P(x)\\frac{dD}{dx}\n\\]\n\nSo **option C is only correct if \\( D(x) \\) is constant**.\n\nOption E:\n\\[\nJ(x) = -\\mu(x)P(x) - D(x)\\frac{dP}{dx}\n\\]\nwhich is not correct.\n\nOption B:\n\\[\nJ(x) = -\\mu(x)P(x) - \\frac{d}{dx}[D(x)P(x)]\n\\]\nwhich has the wrong sign in front of the drift term.\n\nTherefore, the correct answer from the options given, for arbitrary \\( D(x) \\), is **not listed exactly**. The closest correct general form is:\n\n\\[\nJ_{\\text{st}}(x) = \\mu(x) P_{\\text{st}}(x) - \\frac{d}{dx} [D(x) P_{\\text{st}}(x)]\n\\]\n\nBut **none of the options matches this for general \\( D(x) \\) except option J (\"None of the above\")**.\n\nSo, the answer is:\n\n\\[\n\\boxed{J}\n\\]",
    "parsed_option": "J",
    "reward": 0.0,
    "difficulty": 0.25
}
```

## Licensing Information

Data links: ?

### Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
- trafilatura [Apache 2.0 license.](https://trafilatura.readthedocs.io/en/latest/#license)
