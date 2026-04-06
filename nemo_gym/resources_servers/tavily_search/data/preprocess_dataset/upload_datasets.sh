cd /lustre/fsw/portfolios/llmservice/users/rgala/repos/browsecomp-integration-finsih
source .venv/bin/activate

ng_upload_dataset_to_gitlab \
     +dataset_name=tavily_search \
     +version=0.0.2 \
     +input_jsonl_fpath=resources_servers/tavily_search/data/benchmark/browsecomp/browsecomp_test_set.jsonl

ng_upload_dataset_to_gitlab \
     +dataset_name=tavily_search \
     +version=0.0.2 \
     +input_jsonl_fpath=resources_servers/tavily_search/data/benchmark/browsecomp/browsecomp_subset_n_400.jsonl
