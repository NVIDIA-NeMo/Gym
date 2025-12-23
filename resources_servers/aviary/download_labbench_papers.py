import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm


def extract_doi_from_url(url: str) -> str | None:
    patterns = [
        r'doi\.org/(.+)$',
        r'dx\.doi\.org/(.+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_open_access_url(doi: str, email: str) -> dict | None:
    url = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email={email}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return {
                'doi': doi,
                'title': data.get('title', 'Unknown'),
                'is_oa': data.get('is_oa', False),
                'oa_url': data.get('best_oa_location', {}).get('url_for_pdf') if data.get('best_oa_location') else None,
                'oa_landing': data.get('best_oa_location', {}).get('url') if data.get('best_oa_location') else None,
            }
        elif response.status_code == 404:
            return {'doi': doi, 'is_oa': False, 'error': 'DOI not found'}
        else:
            return {'doi': doi, 'is_oa': False, 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'doi': doi, 'is_oa': False, 'error': str(e)}


def download_pdf(url: str, output_path: Path) -> bool:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; LABBenchDownloader/1.0; mailto:research@example.com)'
        }
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
            output_path.write_bytes(response.content)
            return True
        elif response.status_code == 200:
            return False
    except Exception as e:
        print(f"  Download error: {e}")
    return False


def get_all_dois_from_labbench() -> set[str]:
    from datasets import load_dataset

    all_dois = set()

    paper_categories = ['LitQA2', 'SuppQA', 'FigQA', 'TableQA']

    for cat in paper_categories:
        print(f"Loading {cat}...")
        ds = load_dataset('futurehouse/lab-bench', cat, split='train')
        for row in ds:
            sources = row.get('sources', []) or []
            source = row.get('source', None)

            if isinstance(sources, list):
                for s in sources:
                    if s and 'doi' in s.lower():
                        doi = extract_doi_from_url(s)
                        if doi:
                            all_dois.add(doi)

            if source and 'doi' in str(source).lower():
                doi = extract_doi_from_url(source)
                if doi:
                    all_dois.add(doi)

    return all_dois


def main():
    parser = argparse.ArgumentParser(description='Download LAB-Bench papers')
    parser.add_argument('--email', required=True, help='Email for Unpaywall API (required)')
    parser.add_argument('--output-dir', default='./labbench_papers', help='Output directory for papers')
    parser.add_argument('--check-only', action='store_true', help='Only check availability, do not download')
    parser.add_argument('--dois-file', help='File with DOIs (one per line) instead of fetching from dataset')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dois_file:
        with open(args.dois_file) as f:
            dois = set()
            for line in f:
                line = line.strip()
                if line:
                    doi = extract_doi_from_url(line) if 'doi' in line.lower() else line
                    if doi:
                        dois.add(doi)
    else:
        print("Extracting DOIs from LAB-Bench dataset...")
        dois = get_all_dois_from_labbench()

    print(f"\nFound {len(dois)} unique DOIs to process")

    results = {
        'open_access': [],
        'not_open_access': [],
        'downloaded': [],
        'failed': [],
        'errors': [],
    }

    print("\nChecking open access status via Unpaywall...")
    for doi in tqdm(sorted(dois)):
        info = get_open_access_url(doi, args.email)
        time.sleep(0.1)

        if info.get('error'):
            results['errors'].append(info)
            continue

        if info.get('is_oa') and info.get('oa_url'):
            results['open_access'].append(info)

            if not args.check_only:
                safe_doi = doi.replace('/', '_').replace(':', '_')
                pdf_path = output_dir / f"{safe_doi}.pdf"

                if pdf_path.exists():
                    results['downloaded'].append(info)
                elif download_pdf(info['oa_url'], pdf_path):
                    results['downloaded'].append(info)
                    print(f"  Downloaded: {info.get('title', doi)[:60]}...")
                else:
                    results['failed'].append(info)
        else:
            results['not_open_access'].append(info)

    print("\n" + "="*60 + "\n")
    print(f"Total DOIs: {len(dois)}")
    print(f"Open Access (PDF available): {len(results['open_access'])}")
    print(f"Not Open Access: {len(results['not_open_access'])}")
    print(f"Errors: {len(results['errors'])}")

    if not args.check_only:
        print(f"Successfully downloaded: {len(results['downloaded'])}")
        print(f"Failed to download: {len(results['failed'])}")
        print(f"\nPapers saved to: {output_dir}")

    results_file = output_dir / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total': len(dois),
                'open_access': len(results['open_access']),
                'not_open_access': len(results['not_open_access']),
                'downloaded': len(results['downloaded']),
                'failed': len(results['failed']),
                'errors': len(results['errors']),
            },
            'details': results,
        }, f, indent=2)
    print(f"Results saved to: {results_file}")

    if results['not_open_access']:
        print(f"\n{len(results['not_open_access'])} papers require manual download or institutional access. First 10:")
        for info in results['not_open_access'][:10]:
            print(f"  - {info['doi']}")


if __name__ == '__main__':
    main()
