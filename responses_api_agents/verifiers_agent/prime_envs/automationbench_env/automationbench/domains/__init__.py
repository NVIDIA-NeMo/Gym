"""Domain-specific task definitions for AutomationBench."""

from collections.abc import Callable

from datasets import Dataset, concatenate_datasets

from automationbench.domains.finance import get_finance_dataset
from automationbench.domains.hr import get_hr_dataset
from automationbench.domains.marketing import get_marketing_dataset
from automationbench.domains.operations import get_operations_dataset
from automationbench.domains.sales import get_sales_dataset
from automationbench.domains.simple import get_simple_dataset
from automationbench.domains.support import get_support_dataset

# Registry mapping domain names to their dataset loaders
DOMAINS: dict[str, Callable[[], Dataset]] = {
    "sales": get_sales_dataset,
    "marketing": get_marketing_dataset,
    "operations": get_operations_dataset,
    "support": get_support_dataset,
    "finance": get_finance_dataset,
    "hr": get_hr_dataset,
    "simple": get_simple_dataset,
}

PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
DEFAULT_DOMAINS = list(PUBLIC_DOMAINS)

# Meta-domain aliases that expand to multiple domains
DOMAIN_ALIASES: dict[str, list[str]] = {
    "public": PUBLIC_DOMAINS,
}


def get_available_domains() -> list[str]:
    """Return list of available domain names."""
    return list(DOMAINS.keys())


def get_domain_dataset(domain: str) -> Dataset:
    """Get the dataset for a specific domain."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Available: {get_available_domains()}")
    return DOMAINS[domain]()


def get_combined_dataset(domains: list[str]) -> Dataset:
    """Get a combined dataset from multiple domains."""
    if not domains:
        raise ValueError("At least one domain must be specified")
    datasets = [get_domain_dataset(d) for d in domains]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)
