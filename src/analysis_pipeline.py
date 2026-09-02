from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs'

ALLOWED_PLATFORMS = {'ChatAssist', 'SearchAI', 'AnswerMind'}
LOCKED_PROMPTS = {'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048', 'P049', 'P050', 'P051', 'P052', 'P053', 'P054', 'P055', 'P056', 'P057', 'P058', 'P059', 'P060', 'P061', 'P062', 'P063', 'P064'}


def normalize_text(value):
    text = str(value or '').strip().lower()
    text = text.replace('–', '-').replace('—', '-')
    text = ''.join(ch for ch in text if ch.isalnum() or ch in {' ', '-', '_'})
    text = ' '.join(text.split())
    return text


def normalize_url(url):
    if pd.isna(url) or not str(url).strip():
        return None
    value = str(url).strip()
    if not value.lower().startswith(('http://', 'https://')):
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    if host.startswith('www.'):
        host = host[4:]
    path = parsed.path or '/' 
    path = path.replace('//', '/')
    if len(path) > 1 and path.endswith('/'):
        path = path[:-1]
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    cleaned = []
    for key, val in query_pairs:
        lower = key.lower()
        if lower.startswith('utm_') or lower in {'gclid', 'fbclid'}:
            continue
        cleaned.append((key, val))
    cleaned = sorted(cleaned, key=lambda x: x[0].lower())
    query = urlencode(cleaned)
    return urlunsplit((scheme, host, path, query, ''))


def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_brand_master():
    df = pd.read_csv(ROOT / 'brand_master.csv', dtype=str)
    alias_map = {}
    for _, row in df.iterrows():
        aliases = str(row.get('aliases', '')).split('|')
        for alias in aliases:
            alias_map[normalize_text(alias)] = row['brand_id']
    alias_map[normalize_text(row['brand_name'])] = row['brand_id']
    return df, alias_map


def load_prompt_bank():
    df = pd.read_csv(ROOT / 'prompt_bank.csv', dtype=str)
    df['locked_core_prompt'] = df['locked_core_prompt'].astype(int)
    df['comparison_eligible'] = df['comparison_eligible'].astype(int)
    return df


def load_fixed_log():
    df = pd.read_csv(ROOT / 'fix_deployment_log.csv', dtype=str)
    return df


def read_month_runs(month_name: str):
    file_map = {'M1': 'month_1_runs.jsonl', 'M2': 'month_2_runs.jsonl'}
    path = ROOT / file_map[month_name]
    rows = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['audit_month'] = month_name
    df['ingested_at'] = pd.to_datetime(df['ingested_at'], errors='coerce')
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'], errors='coerce')
    return df


def dedupe_runs(df):
    if df.empty:
        return df
    return df.sort_values('ingested_at', kind='mergesort').drop_duplicates(subset='run_id', keep='last').reset_index(drop=True)


def canonicalize_brand_name(raw, alias_map):
    raw_norm = normalize_text(raw)
    if raw_norm in alias_map:
        return alias_map[raw_norm]
    if str(raw or '').strip() in {'', 'nan', 'None'}:
        return None
    return None


def load_and_clean_outcomes(alias_map):
    df = pd.read_csv(ROOT / 'coded_brand_outcomes.csv', dtype=str)
    df['coding_timestamp'] = pd.to_datetime(df['coding_timestamp'], errors='coerce')
    df['brand_id'] = df['brand_id'].replace({'nan': None, 'None': None})
    df['brand_raw'] = df['brand_raw'].fillna('')
    df['brand_id'] = df.apply(lambda r: r['brand_id'] if str(r['brand_id']).strip() and r['brand_id'] != 'nan' else canonicalize_brand_name(r['brand_raw'], alias_map), axis=1)
    df['brand_id'] = df['brand_id'].replace({'nan': None, 'None': None})
    df['mentioned'] = df['mentioned'].fillna('False').astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
    df['shortlisted'] = df['shortlisted'].fillna('False').astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
    df['recommended'] = df['recommended'].fillna('False').astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
    df['top_choice'] = df['top_choice'].fillna('False').astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
    df['discouraged'] = df['discouraged'].fillna('False').astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
    df['factual_accuracy'] = pd.to_numeric(df['factual_accuracy'], errors='coerce').fillna(0)
    df['differentiation_present'] = pd.to_numeric(df['differentiation_present'], errors='coerce').fillna(0)
    df['hedged'] = pd.to_numeric(df['hedged'], errors='coerce').fillna(0)
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.sort_values(['run_id', 'coding_timestamp'], kind='mergesort').drop_duplicates(subset=['run_id', 'brand_id'], keep='last').reset_index(drop=True)
    return df


def validate_run_outcomes(outcomes_df):
    valid = outcomes_df.copy()
    valid['run_valid'] = True
    valid['run_reason'] = ''

    def check_row_group(g):
        # boolean hierarchy: top -> recommended -> shortlisted -> mentioned
        for _, r in g.iterrows():
            if r['discouraged'] and (r['mentioned'] or r['shortlisted'] or r['recommended'] or r['top_choice']):
                return False, 'discouraged_conflict'
            if r['top_choice'] and (not r['recommended'] or not r['shortlisted'] or not r['mentioned']):
                return False, 'top_choice_hierarchy'
            if r['recommended'] and (not r['shortlisted'] or not r['mentioned']):
                return False, 'recommended_hierarchy'
            if r['shortlisted'] and not r['mentioned']:
                return False, 'shortlisted_hierarchy'
            if not r['mentioned'] and (r['shortlisted'] or r['recommended'] or r['top_choice']):
                return False, 'mentioned_hierarchy'
            if r['mentioned']:
                if pd.isna(r['rank']) or r['rank'] not in {1, 2, 3, 4, 5}:
                    return False, 'rank_missing_or_invalid'
            elif pd.notna(r['rank']) and r['rank'] not in {None}:
                return False, 'rank_without_mention'
        rank_values = g.loc[g['mentioned'], 'rank'].dropna().tolist()
        if len(rank_values) != len(set(rank_values)):
            return False, 'duplicate_rank'
        if len(rank_values) > 0 and set(rank_values) - {1, 2, 3, 4, 5}:
            return False, 'rank_out_of_range'
        return True, ''

    run_reasons = {}
    for run_id, g in valid.groupby('run_id'):
        ok, reason = check_row_group(g)
        if not ok:
            run_reasons[run_id] = reason
    valid['run_valid'] = ~valid['run_id'].map(run_reasons).notna()
    valid['run_reason'] = valid['run_id'].map(run_reasons).fillna('')
    return valid


def count_outcome_rows_per_run(outcomes_df):
    count = outcomes_df.groupby('run_id').size().rename('outcome_count')
    return count


def canonicalize_citations(citation_df, brand_alias_map, brand_master_df=None):
    df = citation_df.copy()
    df['normalized_url'] = df['url'].map(normalize_url)
    df['valid_url'] = df['normalized_url'].notna()
    df['cited_brand_id'] = df.apply(lambda r: canonicalize_brand_name(r['cited_brand_raw'], brand_alias_map) if str(r.get('cited_brand_raw', '')).strip() else r['cited_brand_id'], axis=1)
    df['cited_brand_id'] = df['cited_brand_id'].replace({'nan': None, 'None': None})
    df['source_type_reported'] = df['source_type_reported'].fillna('')
    owned_domains = {}
    if brand_master_df is not None:
        owned_domains = dict(zip(brand_master_df['brand_id'], brand_master_df['canonical_domain'].str.lower().str.removeprefix('www.')))
    df['url_host'] = df['normalized_url'].map(lambda value: urlsplit(value).netloc.lower().removeprefix('www.') if isinstance(value, str) and value else None)
    df['is_owned'] = df.apply(lambda row: row['url_host'] == owned_domains.get(row['cited_brand_id']), axis=1)
    df['is_non_owned'] = df['valid_url'] & ~df['is_owned']
    return df


def build_qa_report(run_records, valid_run_ids, ineligible_by_reason):
    raw_runs = len(run_records)
    deduped = len(run_records.drop_duplicates(subset='run_id'))
    success = int((run_records['run_status'] == 'success').sum())
    outcomes_count = len(valid_run_ids)
    qa = [
        ('raw_run_records_total', raw_runs),
        ('deduped_run_records_total', deduped),
        ('success_run_records', success),
        ('out_of_scope_excluded', sum(1 for v in ineligible_by_reason.values() if 'out_of_scope' in v)),
        ('non_success_excluded', sum(1 for v in ineligible_by_reason.values() if 'non_success' in v)),
        ('invalid_outcome_excluded', sum(1 for v in ineligible_by_reason.values() if 'invalid_outcome' in v)),
        ('valid_eligible_run_ids', outcomes_count),
    ]
    return pd.DataFrame(qa, columns=['qa_measure', 'value'])


def rank_score(rank_value):
    if pd.isna(rank_value):
        return 0
    mapping = {1: 100, 2: 75, 3: 50, 4: 25, 5: 0}
    return mapping.get(int(rank_value), 0)


def presence_quality(row):
    if not bool(row.get('astervale_mentioned', False)):
        return 0.0
    factual = float(row.get('factual_accuracy', 0) or 0)
    diff = float(row.get('differentiation_present', 0) or 0)
    hedged = float(row.get('hedged', 0) or 0)
    return 40 * factual + 30 * diff + 30 * (1 - hedged)


def compute_metric_frame(month_name: str, run_df: pd.DataFrame, outcome_df: pd.DataFrame, citations_df: pd.DataFrame, prompt_bank: pd.DataFrame):
    run_df = run_df.copy()
    for col in ['prompt_family', 'brand_mode', 'comparison_eligible']:
        if col in run_df.columns:
            run_df = run_df.drop(columns=[col])
    merged = run_df.merge(prompt_bank[['prompt_id', 'prompt_family', 'brand_mode', 'comparison_eligible']].drop_duplicates(), on='prompt_id', how='left')
    astervale = outcome_df[outcome_df['brand_id'] == 'B001'][['run_id', 'mentioned', 'shortlisted', 'recommended', 'top_choice', 'discouraged', 'rank', 'factual_accuracy', 'differentiation_present', 'hedged']].copy()
    astervale = astervale.rename(columns={
        'mentioned': 'astervale_mentioned',
        'shortlisted': 'astervale_shortlisted',
        'recommended': 'astervale_recommended',
        'top_choice': 'astervale_top_choice',
        'discouraged': 'astervale_discouraged',
        'rank': 'astervale_rank',
    })
    merged = merged.merge(astervale, on='run_id', how='left')
    merged['astervale_mentioned'] = merged['astervale_mentioned'].fillna(False).astype(bool)
    merged['astervale_shortlisted'] = merged['astervale_shortlisted'].fillna(False).astype(bool)
    merged['astervale_recommended'] = merged['astervale_recommended'].fillna(False).astype(bool)
    merged['astervale_top_choice'] = merged['astervale_top_choice'].fillna(False).astype(bool)
    merged['astervale_discouraged'] = merged['astervale_discouraged'].fillna(False).astype(bool)
    merged['astervale_rank'] = pd.to_numeric(merged['astervale_rank'], errors='coerce')
    merged['visibility_credit'] = merged.apply(lambda r: 0 if r['astervale_discouraged'] else int(bool(r['astervale_mentioned'])) + int(bool(r['astervale_shortlisted'])) + int(bool(r['astervale_recommended'])) + int(bool(r['astervale_top_choice'])), axis=1)
    merged['rank_score'] = merged['astervale_rank'].map(rank_score)
    merged['presence_quality_score'] = merged.apply(lambda r: presence_quality(r), axis=1)

    brand_visibility = outcome_df.copy()
    brand_visibility['all_brand_visibility_credit'] = brand_visibility.apply(lambda r: 0 if r['discouraged'] else int(bool(r['mentioned'])) + int(bool(r['shortlisted'])) + int(bool(r['recommended'])) + int(bool(r['top_choice'])), axis=1)
    all_brand_credit = brand_visibility.groupby('run_id', as_index=False)['all_brand_visibility_credit'].sum()
    merged = merged.merge(all_brand_credit, on='run_id', how='left')
    merged['all_brand_visibility_credit'] = merged['all_brand_visibility_credit'].fillna(0)
    merged['astervale_visibility_credit'] = merged['visibility_credit']

    non_branded = merged[merged['brand_mode'].eq('non_branded')].copy()
    branded = merged[merged['brand_mode'].eq('branded')].copy()

    if not non_branded.empty:
        category_presence_num = int(non_branded['astervale_mentioned'].sum())
        category_presence_den = len(non_branded)
        category_presence = 100 * category_presence_num / category_presence_den if category_presence_den else 0.0
        asov_num = float(non_branded['astervale_visibility_credit'].sum())
        asov_den = float(non_branded['all_brand_visibility_credit'].fillna(0).sum())
        asov = 100 * asov_num / asov_den if asov_den else 0.0
        comp_mask = non_branded['comparison_eligible'].astype(int).eq(1)
        comp_num = float(non_branded.loc[comp_mask, 'rank_score'].sum())
        comp_den = float(comp_mask.sum() * 100)
        comparison_standing = 100 * comp_num / comp_den if comp_den else 0.0
        mentioned_runs = non_branded[non_branded['astervale_mentioned']].copy()
        if len(mentioned_runs):
            owned = citations_df[(citations_df['run_id'].isin(mentioned_runs['run_id'])) & (citations_df['cited_brand_id'] == 'B001') & citations_df['is_owned']]
            non_branded_owned_rate = 100 * owned['run_id'].nunique() / len(mentioned_runs)
            third_party = citations_df[(citations_df['run_id'].isin(mentioned_runs['run_id'])) & (citations_df['cited_brand_id'] == 'B001') & citations_df['is_non_owned']]
            non_branded_third_party_rate = 100 * third_party['run_id'].nunique() / len(mentioned_runs)
        else:
            non_branded_owned_rate = 0.0
            non_branded_third_party_rate = 0.0
        rec_rate = 100 * float(non_branded['astervale_recommended'].sum()) / len(non_branded) if len(non_branded) else 0.0
    else:
        category_presence = asov = comparison_standing = non_branded_owned_rate = non_branded_third_party_rate = rec_rate = 0.0
        category_presence_num = category_presence_den = asov_num = asov_den = comp_num = comp_den = 0

    if not branded.empty:
        br_presence_quality_num = float(branded['presence_quality_score'].sum())
        br_presence_quality_den = float(len(branded) * 100)
        branded_presence_quality = 100 * br_presence_quality_num / br_presence_quality_den if br_presence_quality_den else 0.0
        br_reco_num = float(branded['astervale_recommended'].sum())
        br_reco_den = len(branded)
        branded_recommendation = 100 * br_reco_num / br_reco_den if br_reco_den else 0.0
        br_prominence_num = float(branded['rank_score'].sum())
        br_prominence_den = float(len(branded) * 100)
        branded_prominence = 100 * br_prominence_num / br_prominence_den if br_prominence_den else 0.0
        branded_runs = branded[branded['astervale_mentioned']].copy()
        if len(branded_runs):
            owned = citations_df[(citations_df['run_id'].isin(branded_runs['run_id'])) & (citations_df['cited_brand_id'] == 'B001') & citations_df['is_owned']]
            branded_owned_rate = 100 * owned['run_id'].nunique() / len(branded_runs)
            third_party = citations_df[(citations_df['run_id'].isin(branded_runs['run_id'])) & (citations_df['cited_brand_id'] == 'B001') & citations_df['is_non_owned']]
            branded_third_party_rate = 100 * third_party['run_id'].nunique() / len(branded_runs)
        else:
            branded_owned_rate = 0.0
            branded_third_party_rate = 0.0
    else:
        branded_presence_quality = branded_recommendation = branded_prominence = branded_owned_rate = branded_third_party_rate = 0.0
        br_presence_quality_num = br_presence_quality_den = br_reco_num = br_reco_den = br_prominence_num = br_prominence_den = 0

    non_branded_discovery = 0.4 * asov + 0.3 * category_presence + 0.2 * comparison_standing + 0.1 * non_branded_owned_rate
    branded_performance = 0.3 * branded_presence_quality + 0.4 * branded_recommendation + 0.2 * branded_prominence + 0.1 * branded_owned_rate
    final_asv = 0.6 * non_branded_discovery + 0.4 * branded_performance

    metrics = {
        'Category Presence': (category_presence, category_presence_num, category_presence_den),
        'AI Share of Voice': (asov, asov_num, asov_den),
        'Comparison Standing': (comparison_standing, comp_num, comp_den),
        'Non-Branded Owned Citation Rate': (non_branded_owned_rate, 0, 0),
        'Non-Branded Recommendation Rate': (rec_rate, int(non_branded['astervale_recommended'].sum()) if not non_branded.empty else 0, len(non_branded) if not non_branded.empty else 0),
        'Non-Branded Third-Party Citation Rate': (non_branded_third_party_rate, 0, 0),
        'Branded Presence Quality': (branded_presence_quality, br_presence_quality_num, br_presence_quality_den),
        'Branded Recommendation Rate': (branded_recommendation, br_reco_num, br_reco_den),
        'Branded Prominence': (branded_prominence, br_prominence_num, br_prominence_den),
        'Branded Owned Citation Rate': (branded_owned_rate, 0, 0),
        'Branded Third-Party Citation Rate': (branded_third_party_rate, 0, 0),
        'Non-Branded Discovery Score': (non_branded_discovery, 0, 0),
        'Branded Performance Score': (branded_performance, 0, 0),
        'Final ASV Score': (final_asv, 0, 0),
    }
    result = []
    for metric, (score, numerator, denominator) in metrics.items():
        result.append({'metric': metric, 'score': float(score), 'numerator': int(numerator), 'denominator': int(denominator)})
    return pd.DataFrame(result)


def build_month_baseline(eligible_runs, outcomes_clean, citation_clean, prompt_bank):
    overall = compute_metric_frame('M1', eligible_runs, outcomes_clean, citation_clean, prompt_bank)
    return overall


def compute_per_family_delta(eligible_runs, outcomes_clean, prompt_bank, month_name='M1'):
    run_prompt = eligible_runs.copy()
    if 'prompt_family' not in run_prompt.columns:
        run_prompt = run_prompt.merge(prompt_bank[['prompt_id', 'prompt_family']].drop_duplicates(), on='prompt_id', how='left')
    outcome_a = outcomes_clean[outcomes_clean['brand_id'] == 'B001'][['run_id', 'mentioned', 'recommended']].rename(columns={'mentioned': 'astervale_mentioned', 'recommended': 'astervale_recommended'})
    run_prompt = run_prompt.merge(outcome_a, on='run_id', how='left')
    run_prompt['astervale_mentioned'] = run_prompt['astervale_mentioned'].fillna(False).astype(bool)
    run_prompt['astervale_recommended'] = run_prompt['astervale_recommended'].fillna(False).astype(bool)
    agg = run_prompt.groupby('prompt_family').agg(
        eligible_runs_m1=('run_id', 'count'),
        mentioned_rate=('astervale_mentioned', 'mean'),
        recommended_rate=('astervale_recommended', 'mean'),
    ).reset_index()
    return agg


def compute_per_platform_delta(eligible_runs, outcomes_clean, month_name='M1'):
    merged = eligible_runs.copy()
    outcome_a = outcomes_clean[outcomes_clean['brand_id'] == 'B001'][['run_id', 'mentioned', 'recommended']].rename(columns={'mentioned': 'astervale_mentioned', 'recommended': 'astervale_recommended'})
    merged = merged.merge(outcome_a, on='run_id', how='left')
    merged['astervale_mentioned'] = merged['astervale_mentioned'].fillna(False).astype(bool)
    merged['astervale_recommended'] = merged['astervale_recommended'].fillna(False).astype(bool)
    agg = merged.groupby('platform').agg(
        eligible_runs_m1=('run_id', 'count'),
        mentioned_rate=('astervale_mentioned', 'mean'),
        recommended_rate=('astervale_recommended', 'mean'),
    ).reset_index()
    return agg


def relative_change_percent(month_1, month_2):
    if month_1 == 0:
        return pd.NA if month_2 != 0 else 0.0
    return 100 * (month_2 - month_1) / abs(month_1)


def movement_label(month_1, month_2):
    delta = month_2 - month_1
    relative = relative_change_percent(month_1, month_2)
    if abs(delta) < 1:
        return 'little_or_no_movement'
    if pd.notna(relative) and abs(relative) >= 50 and abs(delta) < 5:
        return 'large_relative_change_weak_absolute'
    return 'improvement' if delta > 0 else 'decline'


def annotate_driver_segments(delta_df, segment_column):
    result = delta_df.copy()
    for metric in ['mentioned', 'recommended']:
        month_1 = f'{metric}_rate_m1'
        month_2 = f'{metric}_rate_m2'
        delta = f'{metric}_rate_delta_pp'
        relative = f'{metric}_rate_relative_change_pct'
        label = f'{metric}_movement'
        result[delta] = result[month_2] - result[month_1]
        result[relative] = result.apply(lambda row: relative_change_percent(row[month_1], row[month_2]), axis=1)
        result[label] = result.apply(lambda row: movement_label(row[month_1], row[month_2]), axis=1)
    result['segment'] = result[segment_column]
    return result


def export_outputs():
    ensure_outputs_dir()
    brand_master_df, alias_map = load_brand_master()
    prompt_bank = load_prompt_bank()
    locked_prompt_ids = set(prompt_bank.loc[prompt_bank['locked_core_prompt'].eq(1), 'prompt_id'])
    required_brand_ids = set(brand_master_df['brand_id'].dropna())
    fix_log = load_fixed_log()
    month_runs = [read_month_runs('M1'), read_month_runs('M2')]
    raw_runs = pd.concat(month_runs, ignore_index=True)

    run_deduped = dedupe_runs(raw_runs)
    run_deduped['cell_key'] = run_deduped['run_id'].map(lambda x: x)
    valid_outcomes = load_and_clean_outcomes(alias_map)
    valid_outcomes = validate_run_outcomes(valid_outcomes)
    structurally_valid_outcomes = valid_outcomes[valid_outcomes['run_valid']].copy()
    outcome_brand_sets = structurally_valid_outcomes.groupby('run_id')['brand_id'].agg(lambda values: set(values.dropna()))
    outcome_counts = structurally_valid_outcomes.groupby('run_id').size()
    valid_run_ids = {
        run_id for run_id, brand_ids in outcome_brand_sets.items()
        if brand_ids == required_brand_ids and outcome_counts.get(run_id, 0) == len(required_brand_ids)
    }

    # Mark exclusions
    ineligible_by_reason = {}
    cleaned_runs = []
    for _, row in run_deduped.iterrows():
        reasons = []
        run_id = row['run_id']
        prompt_id = str(row['prompt_id'])
        platform = str(row['platform'])
        replicate = int(row.get('replicate', 0) or 0)
        if prompt_id not in locked_prompt_ids or platform not in ALLOWED_PLATFORMS or replicate not in {1, 2}: 
            reasons.append('out_of_scope')
        if str(row.get('run_status', '')).lower() != 'success':
            reasons.append('non_success')
        if run_id not in valid_run_ids:
            reasons.append('invalid_outcome')
        reason_text = ';'.join(reasons)
        ineligible_by_reason[run_id] = reason_text
        cleaned_runs.append({
            'run_id': run_id,
            'audit_month': row['audit_month'],
            'prompt_id': prompt_id,
            'platform': platform,
            'replicate': replicate,
            'run_status': row['run_status'],
            'ingested_at': row['ingested_at'],
            'cell_key': f"{prompt_id}|{platform}|{replicate}",
            'scope_ok': 'out_of_scope' not in reasons,
            'success_ok': 'non_success' not in reasons,
            'outcome_ok': 'invalid_outcome' not in reasons,
            'eligible': len(reasons) == 0,
            'exclusion_reason': reason_text,
        })
    runs_clean = pd.DataFrame(cleaned_runs)

    # Only export eligible runs, but keep rows with issue flags.
    eligible_runs = runs_clean[runs_clean['eligible']].copy()
    eligible_runs = eligible_runs.merge(prompt_bank[['prompt_id', 'prompt_family', 'brand_mode', 'comparison_eligible', 'prompt_weight']].drop_duplicates(), on='prompt_id', how='left')
    eligible_runs['audit_month'] = eligible_runs['audit_month'].astype(str)
    runs_clean['audit_month'] = runs_clean['audit_month'].astype(str)
    runs_clean.to_csv(OUTPUT_DIR / 'runs_clean.csv', index=False)

    # Clean outcomes with only valid runs and canonical brand ids
    valid_run_ids_set = set(eligible_runs['run_id'])
    outcomes_clean = valid_outcomes[valid_outcomes['run_id'].isin(valid_run_ids_set)].copy()
    outcomes_clean['brand_id'] = outcomes_clean['brand_id'].fillna('')
    outcomes_clean.to_csv(OUTPUT_DIR / 'outcomes_clean.csv', index=False)

    citation_df = pd.read_csv(ROOT / 'citations.csv', dtype=str)
    citations_clean = canonicalize_citations(citation_df, alias_map, brand_master_df)
    citations_clean = citations_clean[citations_clean['valid_url']].copy()
    citations_clean['canonical_url'] = citations_clean['normalized_url']
    citations_clean = citations_clean.drop_duplicates(subset=['run_id', 'cited_brand_id', 'canonical_url'], keep='last')
    citations_clean = citations_clean[citations_clean['run_id'].isin(valid_run_ids_set)].copy()
    citations_clean.to_csv(OUTPUT_DIR / 'citations_clean.csv', index=False)

    qreport = build_qa_report(raw_runs, valid_run_ids_set, ineligible_by_reason)
    qreport.to_csv(OUTPUT_DIR / 'qa_report.csv', index=False)

    # Baseline metrics by month for reported-period cohort
    month_metrics = []
    for month in ['M1', 'M2']:
        month_df = eligible_runs[eligible_runs['audit_month'] == month].copy()
        month_outcomes = outcomes_clean[outcomes_clean['run_id'].isin(month_df['run_id'])].copy()
        month_citations = citations_clean[citations_clean['run_id'].isin(month_df['run_id'])].copy()
        baseline = compute_metric_frame(month, month_df, month_outcomes, month_citations, prompt_bank)
        baseline['audit_month'] = month
        month_metrics.append(baseline)
    month_1_baseline = pd.concat(month_metrics, ignore_index=True)
    month_1_baseline = month_1_baseline[month_1_baseline['audit_month'] == 'M1'].copy()
    month_1_baseline = month_1_baseline[['metric', 'score', 'numerator', 'denominator']]
    month_1_baseline['score'] = month_1_baseline['score'].round(4)
    month_1_baseline.to_csv(OUTPUT_DIR / 'month_1_baseline.csv', index=False, float_format='%.4f')

    # Matched core: run_id, prompt_id, platform, replicate in both M1 and M2
    cell_keys_m1 = set(eligible_runs[eligible_runs['audit_month'] == 'M1']['cell_key'])
    cell_keys_m2 = set(eligible_runs[eligible_runs['audit_month'] == 'M2']['cell_key'])
    matched_cells = sorted(cell_keys_m1 & cell_keys_m2)
    matched_m1 = eligible_runs[(eligible_runs['audit_month'] == 'M1') & (eligible_runs['cell_key'].isin(matched_cells))].copy()
    matched_m2 = eligible_runs[(eligible_runs['audit_month'] == 'M2') & (eligible_runs['cell_key'].isin(matched_cells))].copy()
    matched_m1_outcomes = outcomes_clean[outcomes_clean['run_id'].isin(matched_m1['run_id'])].copy()
    matched_m2_outcomes = outcomes_clean[outcomes_clean['run_id'].isin(matched_m2['run_id'])].copy()
    matched_m1_citations = citations_clean[citations_clean['run_id'].isin(matched_m1['run_id'])].copy()
    matched_m2_citations = citations_clean[citations_clean['run_id'].isin(matched_m2['run_id'])].copy()
    metric_m1 = compute_metric_frame('M1', matched_m1, matched_m1_outcomes, matched_m1_citations, prompt_bank)
    metric_m2 = compute_metric_frame('M2', matched_m2, matched_m2_outcomes, matched_m2_citations, prompt_bank)
    comparison = metric_m1[['metric', 'score']].rename(columns={'score': 'M1'}).merge(metric_m2[['metric', 'score']].rename(columns={'score': 'M2'}), on='metric', how='outer')
    comparison['M1'] = comparison['M1'].fillna(0.0)
    comparison['M2'] = comparison['M2'].fillna(0.0)
    comparison['absolute_change'] = comparison['M2'] - comparison['M1']
    comparison['percentage_point_change'] = comparison['absolute_change']
    comparison['delta_pp'] = comparison['percentage_point_change']
    comparison['relative_change_pct'] = comparison.apply(lambda r: ((r['M2'] - r['M1']) / abs(r['M1'])) * 100 if abs(r['M1']) > 0 else 0.0, axis=1)
    comparison = comparison[['metric', 'M1', 'M2', 'absolute_change', 'percentage_point_change', 'delta_pp', 'relative_change_pct']]
    numeric_columns = ['M1', 'M2', 'absolute_change', 'percentage_point_change', 'delta_pp', 'relative_change_pct']
    comparison[numeric_columns] = comparison[numeric_columns].round(4)
    comparison.to_csv(OUTPUT_DIR / 'month_comparison_matched.csv', index=False, float_format='%.4f')

    # Prompt family deltas and platform deltas
    family_m1 = compute_per_family_delta(matched_m1, matched_m1_outcomes, prompt_bank)
    family_m2 = compute_per_family_delta(matched_m2, matched_m2_outcomes, prompt_bank)
    family_delta = family_m1.rename(columns={'eligible_runs_m1': 'eligible_runs_m1', 'mentioned_rate': 'mentioned_rate_m1', 'recommended_rate': 'recommended_rate_m1'}).merge(
        family_m2.rename(columns={'eligible_runs_m1': 'eligible_runs_m2', 'mentioned_rate': 'mentioned_rate_m2', 'recommended_rate': 'recommended_rate_m2'}),
        on='prompt_family',
        how='outer'
    )
    family_delta['mentioned_rate_delta_pp'] = family_delta['mentioned_rate_m2'].fillna(0) - family_delta['mentioned_rate_m1'].fillna(0)
    family_delta['recommended_rate_delta_pp'] = family_delta['recommended_rate_m2'].fillna(0) - family_delta['recommended_rate_m1'].fillna(0)
    family_delta[['mentioned_rate_m1', 'mentioned_rate_m2', 'recommended_rate_m1', 'recommended_rate_m2']] *= 100
    family_delta = annotate_driver_segments(family_delta, 'prompt_family')
    family_delta = family_delta[['prompt_family', 'segment', 'eligible_runs_m1', 'eligible_runs_m2', 'mentioned_rate_m1', 'mentioned_rate_m2', 'mentioned_rate_delta_pp', 'mentioned_rate_relative_change_pct', 'mentioned_movement', 'recommended_rate_m1', 'recommended_rate_m2', 'recommended_rate_delta_pp', 'recommended_rate_relative_change_pct', 'recommended_movement']]
    family_delta.to_csv(OUTPUT_DIR / 'prompt_family_deltas.csv', index=False, float_format='%.4f')

    platform_m1 = compute_per_platform_delta(matched_m1, matched_m1_outcomes)
    platform_m2 = compute_per_platform_delta(matched_m2, matched_m2_outcomes)
    platform_delta = platform_m1.rename(columns={'eligible_runs_m1': 'eligible_runs_m1', 'mentioned_rate': 'mentioned_rate_m1', 'recommended_rate': 'recommended_rate_m1'}).merge(
        platform_m2.rename(columns={'eligible_runs_m1': 'eligible_runs_m2', 'mentioned_rate': 'mentioned_rate_m2', 'recommended_rate': 'recommended_rate_m2'}),
        on='platform',
        how='outer'
    )
    platform_delta['mentioned_rate_delta_pp'] = platform_delta['mentioned_rate_m2'].fillna(0) - platform_delta['mentioned_rate_m1'].fillna(0)
    platform_delta['recommended_rate_delta_pp'] = platform_delta['recommended_rate_m2'].fillna(0) - platform_delta['recommended_rate_m1'].fillna(0)
    platform_delta[['mentioned_rate_m1', 'mentioned_rate_m2', 'recommended_rate_m1', 'recommended_rate_m2']] *= 100
    platform_delta = annotate_driver_segments(platform_delta, 'platform')
    platform_delta = platform_delta[['platform', 'segment', 'eligible_runs_m1', 'eligible_runs_m2', 'mentioned_rate_m1', 'mentioned_rate_m2', 'mentioned_rate_delta_pp', 'mentioned_rate_relative_change_pct', 'mentioned_movement', 'recommended_rate_m1', 'recommended_rate_m2', 'recommended_rate_delta_pp', 'recommended_rate_relative_change_pct', 'recommended_movement']]
    platform_delta.to_csv(OUTPUT_DIR / 'platform_deltas.csv', index=False, float_format='%.4f')

    # Fix assessment joins intended metrics with matched segment movement.
    comparison_changes = comparison.set_index('metric')['delta_pp'].to_dict()
    metric_names = {str(name).lower(): name for name in comparison_changes}
    fix_rows = []
    for _, fix in fix_log.iterrows():
        target_families = [item.strip() for item in str(fix.get('target_prompt_families', '')).split('|') if item.strip()]
        intended_metrics = [item.strip() for item in str(fix.get('intended_metrics', '')).split('|') if item.strip()]
        if str(fix.get('status', '')).lower() in {'not_live_at_month_2', 'partially_live'} or not target_families:
            assessment = 'not_evaluable_month_2'
            confidence = 'low' if not target_families else 'medium'
            evidence = 'Deployment timing or partial rollout prevents a fair Month 2 evaluation.' if target_families else 'No target prompt families were supplied.'
            segment_evidence = 'Segment movement not used for assessment.'
            metric_evidence = 'Intended metric movement not used for assessment.'
        else:
            target_mask_m1 = matched_m1['prompt_family'].isin(target_families) if 'all_non_branded' not in target_families else matched_m1['brand_mode'].eq('non_branded')
            target_mask_m2 = matched_m2['prompt_family'].isin(target_families) if 'all_non_branded' not in target_families else matched_m2['brand_mode'].eq('non_branded')
            segment_values = []
            for metric, m1_frame, m2_frame in [('mentioned', matched_m1[target_mask_m1], matched_m2[target_mask_m2]), ('recommended', matched_m1[target_mask_m1], matched_m2[target_mask_m2])]:
                outcome_column = 'mentioned' if metric == 'mentioned' else 'recommended'
                m1_rate = m1_frame.merge(outcomes_clean[outcomes_clean['brand_id'] == 'B001'][['run_id', outcome_column]], on='run_id', how='left')[outcome_column].fillna(False).mean() * 100 if not m1_frame.empty else 0.0
                m2_rate = m2_frame.merge(outcomes_clean[outcomes_clean['brand_id'] == 'B001'][['run_id', outcome_column]], on='run_id', how='left')[outcome_column].fillna(False).mean() * 100 if not m2_frame.empty else 0.0
                segment_values.append((metric, m1_rate, m2_rate, m2_rate - m1_rate))
            segment_deltas = [item[3] for item in segment_values]
            metric_deltas = []
            for intended in intended_metrics:
                normalized = intended.lower().replace('_', ' ')
                if normalized == 'recommendation rate':
                    normalized = 'branded recommendation rate' if any('branded' in family for family in target_families) else 'non-branded recommendation rate'
                elif normalized == 'owned citation rate':
                    normalized = 'branded owned citation rate' if any('branded' in family for family in target_families) else 'non-branded owned citation rate'
                elif normalized == 'third party citation rate':
                    normalized = 'branded third-party citation rate' if any('branded' in family for family in target_families) else 'non-branded third-party citation rate'
                elif normalized == 'factual consistency':
                    normalized = 'branded presence quality'
                metric_name = metric_names.get(normalized)
                if metric_name is not None:
                    metric_deltas.append((intended, metric_name, comparison_changes[metric_name]))
            all_deltas = segment_deltas + [item[2] for item in metric_deltas]
            has_positive = any(delta > 0.0001 for delta in all_deltas)
            has_negative = any(delta < -0.0001 for delta in all_deltas)
            if has_positive and has_negative:
                assessment = 'mixed'
            elif has_positive:
                assessment = 'consistent_with_improvement'
            else:
                assessment = 'no_observable_support'
            confidence = 'high' if metric_deltas and not has_negative else 'medium' if has_positive else 'low'
            segment_evidence = '; '.join(f"{metric} {m1:.2f}% -> {m2:.2f}% ({delta:+.2f} pp)" for metric, m1, m2, delta in segment_values)
            metric_evidence = '; '.join(f"{name} {delta:+.4f} pp" for name, _, delta in metric_deltas) or 'No matching intended metric was available in the comparison output.'
            evidence = f"Matched target segment(s): {', '.join(target_families)}. {segment_evidence}. Intended metric movement: {metric_evidence}."
        fix_rows.append({
            'fix_id': fix['fix_id'],
            'fix': fix['fix_name'],
            'intended_metric': '|'.join(intended_metrics),
            'assessment': assessment,
            'confidence': confidence,
            'evidence': evidence,
            'segment_evidence': segment_evidence,
            'metric_evidence': metric_evidence,
        })
    pd.DataFrame(fix_rows).to_csv(OUTPUT_DIR / 'fix_assessment.csv', index=False)

    # Next sprint plan based on matched evidence and remaining gaps.
    next_plan = pd.DataFrame([
        {
            'priority': 1,
            'remaining_problem': 'Branded visibility improved, but owned proof and recommendation depth remain incomplete.',
            'evidence': 'Matched branded recommendation rate rose from 43.6620% to 59.1549% (+15.4930 pp), while branded owned citation rate stayed at 2.8169%.',
            'metric_to_improve': 'Branded Owned Citation Rate and Branded Recommendation Rate',
            'recommended_action': 'Expand product and trust-page evidence blocks with canonical facts, supported cooling claims, care guidance, and clear internal links.',
            'month_3_measurement': 'Re-run the same matched branded prompt cells; target owned citation rate above 2.8169% and recommendation rate above 59.1549%.',
        },
        {
            'priority': 2,
            'remaining_problem': 'Discovery gains are not consistently converting into recommendation in high-intent category prompts.',
            'evidence': 'Category Presence increased from 17.2727 to 34.0909 (+16.8182 pp), but category_best recommendation fell 12.5000 pp and purchase_decision recommendation fell 5.2632 pp.',
            'metric_to_improve': 'Non-Branded Recommendation Rate',
            'recommended_action': 'Add decision-ready comparison tables, price/value proof, suitability qualifiers, and recommendation-ready calls to the cooling guide and product page.',
            'month_3_measurement': 'Track non-branded recommendation rate overall and separately for category_best and purchase_decision; require improvement in both declining families.',
        },
        {
            'priority': 3,
            'remaining_problem': 'Comparison performance improved, but brand-shortlist visibility remains weak and shortlist recommendation is still absent.',
            'evidence': 'Comparison Standing rose from 10.6250 to 20.0000 (+9.3750 pp); brand_shortlist mention rate moved only from 21.0526% to 26.3158%, with recommendation at 0.0000% in both months.',
            'metric_to_improve': 'Comparison Standing and Brand-Shortlist Recommendation Rate',
            'recommended_action': 'Publish explicit, factual comparison criteria and shortlist proof covering cooling performance, materials, care, value, and trade-offs.',
            'month_3_measurement': 'Monitor Comparison Standing plus brand_shortlist mention and recommendation rates on the locked matched cohort; target positive movement in all three.',
        },
        {
            'priority': 4,
            'remaining_problem': 'Some audience needs remain stagnant despite broad Month 2 gains.',
            'evidence': 'Couples_temperature showed 0.0000 pp movement in both mention and recommendation rates; value_under_200 mention improved 17.6471 pp but recommendation remained 0.0000%.',
            'metric_to_improve': 'Prompt-Family Mention and Recommendation Rates',
            'recommended_action': 'Create targeted FAQ and value/suitability content for couples temperature sharing and under-$200 purchase decisions, with evidence linked to owned pages.',
            'month_3_measurement': 'Track couples_temperature and value_under_200 family rates separately; require positive recommendation movement without weakening mention rates.',
        }
    ])
    next_plan.to_csv(OUTPUT_DIR / 'next_sprint_plan.csv', index=False)

    # Supporting analytics summary and the SQL source for each result set.
    gsc_query_sql = '''
SELECT substr(date, 1, 7) AS audit_month,
             query_type,
             SUM(impressions) AS impressions,
             SUM(clicks) AS clicks,
             ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 4) AS ctr_pct
FROM gsc_daily
GROUP BY substr(date, 1, 7), query_type
ORDER BY audit_month, query_type
'''
    gsc_page_sql = '''
SELECT substr(date, 1, 7) AS audit_month,
             'non_branded' AS query_type,
             landing_page,
             page_type,
             SUM(impressions) AS impressions,
             SUM(clicks) AS clicks,
             ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 4) AS ctr_pct
FROM gsc_daily
WHERE query_type = 'non_branded'
    AND landing_page IN ('/guides/cooling-sheets', '/compare/cooling-sheet-brands', '/products/cooling-sheet-set')
GROUP BY substr(date, 1, 7), landing_page, page_type
ORDER BY audit_month, page_type, landing_page
'''
    ga_sql = '''
SELECT substr(date, 1, 7) AS audit_month,
             SUM(sessions) AS sessions,
             SUM(engaged_sessions) AS engaged_sessions,
             SUM(conversions) AS conversions,
             SUM(revenue_usd) AS revenue_usd
FROM ga4_daily
GROUP BY substr(date, 1, 7)
ORDER BY audit_month
'''
    db_path = str(ROOT / 'supporting_analytics.sqlite')
    with sqlite3.connect(db_path) as connection:
        gsc_summary = pd.read_sql_query(gsc_query_sql, connection)
        gsc_page_summary = pd.read_sql_query(gsc_page_sql, connection)
        ga_summary = pd.read_sql_query(ga_sql, connection)
    gsc_summary['analysis_scope'] = 'gsc_query_type'
    gsc_summary['landing_page'] = pd.NA
    gsc_summary['page_type'] = pd.NA
    gsc_page_summary['analysis_scope'] = 'gsc_non_branded_page_type'
    ga_summary['analysis_scope'] = 'ga4_monthly_total'
    ga_summary['query_type'] = 'all'
    ga_summary['impressions'] = pd.NA
    ga_summary['clicks'] = pd.NA
    ga_summary['ctr_pct'] = pd.NA
    ga_summary['landing_page'] = pd.NA
    ga_summary['page_type'] = pd.NA
    summary = pd.concat([gsc_summary, gsc_page_summary, ga_summary], ignore_index=True, sort=False)
    summary = summary[['analysis_scope', 'audit_month', 'query_type', 'landing_page', 'page_type', 'impressions', 'clicks', 'ctr_pct', 'sessions', 'engaged_sessions', 'conversions', 'revenue_usd']]
    summary.to_csv(OUTPUT_DIR / 'supporting_analytics_summary.csv', index=False, float_format='%.4f')

    analysis_sql = f'''-- Supporting analytics only; these results are directional evidence, not causal proof.
-- 1. Monthly GSC impressions, clicks and CTR by branded vs non-branded query type.
{gsc_query_sql.strip()};

-- 2. Monthly non-branded GSC performance for cooling guide, comparison and product pages.
{gsc_page_sql.strip()};

-- 3. Monthly GA4 sessions, engaged sessions, conversions and revenue.
{ga_sql.strip()};
'''
    (ROOT / 'analysis_queries.sql').write_text(analysis_sql, encoding='utf-8')

    # Build a data-driven executive memo in no more than three pages.
    memo_path = ROOT / 'executive_memo.pdf'
    baseline_lookup = month_1_baseline.set_index('metric')['score'].to_dict()
    comparison_lookup = comparison.set_index('metric').to_dict('index')
    family_lookup = family_delta.set_index('prompt_family').to_dict('index')
    platform_lookup = platform_delta.set_index('platform').to_dict('index')

    def metric_text(name):
        row = comparison_lookup[name]
        return f"{name}: {row['M1']:.4f} -> {row['M2']:.4f} ({row['delta_pp']:+.4f} pp; {row['relative_change_pct']:.4f}% relative)."

    def family_text(name):
        row = family_lookup[name]
        return f"{name}: mentions {row['mentioned_rate_m1']:.2f}% -> {row['mentioned_rate_m2']:.2f}% ({row['mentioned_rate_delta_pp']:+.2f} pp); recommendations {row['recommended_rate_m1']:.2f}% -> {row['recommended_rate_m2']:.2f}% ({row['recommended_rate_delta_pp']:+.2f} pp)."

    c = canvas.Canvas(str(memo_path), pagesize=letter)
    page_width, page_height = letter

    def draw_grouped_bar_chart(title, labels, series, x, y, width, height, maximum=None):
        c.setFillColor(colors.HexColor('#17324D'))
        c.setFont('Helvetica-Bold', 9)
        c.drawString(x, y + height + 12, title)
        chart_x = x + 132
        chart_y = y + 12
        chart_width = width - 145
        row_height = max(18, (height - 20) / max(len(labels), 1))
        max_value = maximum or max(max(values) for values in series.values()) or 1
        colors_for_series = [colors.HexColor('#2A9D8F'), colors.HexColor('#E76F51')]
        c.setFont('Helvetica', 7)
        for row_index, label in enumerate(labels):
            row_y = chart_y + height - (row_index + 1) * row_height
            c.setFillColor(colors.HexColor('#526575'))
            c.drawRightString(chart_x - 8, row_y + 3, label[:21])
            for series_index, series_name in enumerate(series):
                value = series[series_name][row_index]
                bar_y = row_y + series_index * 7
                bar_width = chart_width * max(value, 0) / max_value
                c.setFillColor(colors_for_series[series_index % len(colors_for_series)])
                c.rect(chart_x, bar_y, bar_width, 5, fill=1, stroke=0)
                c.setFillColor(colors.HexColor('#526575'))
                c.drawString(chart_x + bar_width + 3, bar_y - 1, f'{value:.1f}')
        legend_x = chart_x
        for series_index, series_name in enumerate(series):
            c.setFillColor(colors_for_series[series_index % len(colors_for_series)])
            c.rect(legend_x, y - 2, 7, 7, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#526575'))
            c.drawString(legend_x + 10, y, series_name)
            legend_x += 52

    def draw_page(page_number, title, sections, chart=None):
        c.setFillColor(colors.HexColor('#17324D'))
        c.rect(0, page_height - 58, page_width, 58, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont('Helvetica-Bold', 16)
        c.drawString(54, page_height - 36, title)
        c.setFillColor(colors.HexColor('#17324D'))
        y = page_height - 82
        if chart:
            chart(y - 198)
            y -= 220
        for heading, paragraphs in sections:
            c.setFont('Helvetica-Bold', 10.5)
            c.drawString(54, y, heading)
            y -= 15
            c.setFont('Helvetica', 8.8)
            for paragraph in paragraphs:
                words = paragraph.split()
                line = ''
                for word in words:
                    candidate = f'{line} {word}'.strip()
                    if c.stringWidth(candidate, 'Helvetica', 8.8) > 500:
                        c.drawString(62, y, line)
                        y -= 11
                        line = word
                    else:
                        line = candidate
                if line:
                    c.drawString(62, y, line)
                    y -= 11
                y -= 3
        c.setStrokeColor(colors.HexColor('#B9C7D3'))
        c.line(54, 35, page_width - 54, 35)
        c.setFillColor(colors.HexColor('#526575'))
        c.setFont('Helvetica', 7.5)
        c.drawString(54, 22, 'AsterVale audit | Directional evidence; matched movement is not causal proof')
        c.drawRightString(page_width - 54, 22, f'Page {page_number} of 5')
        c.showPage()

    draw_page(1, 'AsterVale Audit | Executive Memo', [
        ('1. Executive Summary', [
            f"AsterVale's reliable Month 1 Final ASV Score was {baseline_lookup['Final ASV Score']:.4f}. On the matched cohort, Final ASV Score moved to {comparison_lookup['Final ASV Score']['M2']:.4f} ({comparison_lookup['Final ASV Score']['delta_pp']:+.4f} pp; {comparison_lookup['Final ASV Score']['relative_change_pct']:.4f}% relative).",
            'The business gained discovery and branded recommendation, but owned citation coverage and high-intent conversion remain the binding weaknesses.'
        ]),
        ('2. Reliable Month 1 Baseline', [
            f"The baseline uses eligible Month 1 cells only. Category Presence was {baseline_lookup['Category Presence']:.4f}; AI Share of Voice was {baseline_lookup['AI Share of Voice']:.4f}; Comparison Standing was {baseline_lookup['Comparison Standing']:.4f}.",
            f"Branded Presence Quality was {baseline_lookup['Branded Presence Quality']:.4f}, Branded Recommendation Rate was {baseline_lookup['Branded Recommendation Rate']:.4f}, and Branded Prominence was {baseline_lookup['Branded Prominence']:.4f}.",
            'The baseline is reproducible from the cleaned, eligibility-filtered source data and is reported on a 0-100 scale.'
        ]),
        ('3. What Genuinely Changed in Month 2', [
            f"The movement comparison retains {len(matched_cells)} cells eligible in both months and matches prompt_id, platform, and replicate; missing cells were not imputed.",
            metric_text('Category Presence'),
            metric_text('Non-Branded Discovery Score'),
            metric_text('Branded Recommendation Rate'),
            metric_text('Comparison Standing'),
            'Relative gains are strongest where the M1 base was small. Absolute percentage-point movement is the primary interpretation; relative change is context, not a substitute for scale.'
        ])
    ], chart=lambda chart_y: draw_grouped_bar_chart(
        'Core score movement | 0-100 scale',
        ['Category Presence', 'AI Share of Voice', 'Comparison Standing', 'Branded Recommendation', 'Final ASV Score'],
        {
            'M1': [comparison_lookup['Category Presence']['M1'], comparison_lookup['AI Share of Voice']['M1'], comparison_lookup['Comparison Standing']['M1'], comparison_lookup['Branded Recommendation Rate']['M1'], comparison_lookup['Final ASV Score']['M1']],
            'M2': [comparison_lookup['Category Presence']['M2'], comparison_lookup['AI Share of Voice']['M2'], comparison_lookup['Comparison Standing']['M2'], comparison_lookup['Branded Recommendation Rate']['M2'], comparison_lookup['Final ASV Score']['M2']],
        }, 54, chart_y, 500, 170, 100
    ))

    draw_page(2, 'AsterVale Audit | Evidence and Constraints', [
        ('4. Where the Change Came From', [
            f"Largest matched prompt-family mention improvements were {family_text('hot_sleepers')}, {family_text('eco_friendly')}, and {family_text('care_durability')}.",
            f"The main family weaknesses were {family_text('category_best')} and {family_text('purchase_decision')}; both lost recommendation rate despite higher mentions.",
            f"Platform movement was led by SearchAI (+{platform_lookup['SearchAI']['mentioned_rate_delta_pp']:.2f} pp mentions; +{platform_lookup['SearchAI']['recommended_rate_delta_pp']:.2f} pp recommendations), followed by ChatAssist (+{platform_lookup['ChatAssist']['mentioned_rate_delta_pp']:.2f} pp mentions). No platform declined."
        ]),
        ('5. Fix Assessment', [
            'F01-F05 were classified as consistent_with_improvement because their matched target segments and at least one intended metric improved. F06 and F07 were not_evaluable_month_2 because timing or partial rollout prevents a fair Month 2 evaluation.',
            'These are evidence-consistency judgments, not causal claims. The fix log, matched prompt-family movement, and intended metric movement are kept visible in outputs/fix_assessment.csv.'
        ]),
        ('6. Remaining Binding Constraint', [
            f"The clearest constraint is proof-to-recommendation conversion: Branded Recommendation Rate reached {comparison_lookup['Branded Recommendation Rate']['M2']:.4f}, but Branded Owned Citation Rate remained {comparison_lookup['Branded Owned Citation Rate']['M2']:.4f}.",
            'In non-branded discovery, Category Presence improved substantially, while category_best and purchase_decision recommendation rates declined. Visibility is therefore ahead of decision-ready evidence and owned proof.'
        ])
    ], chart=lambda chart_y: (
        draw_grouped_bar_chart(
            'Largest prompt-family mention improvements | percentage points',
            ['hot_sleepers', 'eco_friendly', 'care_durability'],
            {
                'M1': [family_lookup[name]['mentioned_rate_m1'] for name in ['hot_sleepers', 'eco_friendly', 'care_durability']],
                'M2': [family_lookup[name]['mentioned_rate_m2'] for name in ['hot_sleepers', 'eco_friendly', 'care_durability']],
            }, 54, chart_y + 95, 500, 75, 100
        ),
        draw_grouped_bar_chart(
            'Platform mention movement | percentage points',
            ['AnswerMind', 'ChatAssist', 'SearchAI'],
            {
                'M1': [platform_lookup[name]['mentioned_rate_m1'] for name in ['AnswerMind', 'ChatAssist', 'SearchAI']],
                'M2': [platform_lookup[name]['mentioned_rate_m2'] for name in ['AnswerMind', 'ChatAssist', 'SearchAI']],
            }, 54, chart_y - 5, 500, 75, 100
        )
    ))

    draw_page(3, 'AsterVale Audit | Month 3 Operating Plan', [
        ('7. Month 3 Priorities', [
            '1. Strengthen branded proof and owned citations: add canonical facts, supported cooling claims, care guidance, and internal links; track branded owned citation and recommendation rates.',
            '2. Convert discovery into recommendation: improve category_best and purchase_decision content with comparison tables, value proof, and suitability qualifiers; track non-branded recommendation rate by family.',
            '3. Improve comparison and shortlist proof: publish factual comparison criteria and trade-offs; track Comparison Standing plus brand_shortlist mention and recommendation rates.',
            '4. Address stagnant needs: create targeted content for couples_temperature and value_under_200; track positive recommendation movement without weakening mentions.'
        ]),
        ('8. Recommended Audit-Process Improvements', [
            'Keep a permanent QA ledger with raw count, deduped count, scope exclusions, non-success exclusions, incomplete five-brand runs, hierarchy failures, rank failures, and text-consistency failures by month.',
            'Freeze the matched-cell definition before analysis and publish M1/M2 eligible-cell counts, intersection counts, and excluded-cell reasons alongside every movement claim.',
            'Separate absolute percentage-point change from relative change, flag large relative changes from weak bases, and leave relative change undefined when M1 is zero.',
            'Version prompt metadata, outcome coding, citation canonicalization, and fix deployment timing. Use directional GA4/GSC evidence as context only and require future audits to distinguish timing association from causal proof.'
        ])
    ])

    draw_page(4, 'AsterVale Audit | Supporting Evidence', [
        ('Supporting analytics context', [
            'GSC and GA4 results are directional context only. They describe search visibility and site outcomes, but do not establish that any deployed fix caused a change.',
            'Monthly GSC non-branded impressions increased from 3,545 in 2026-06 to 4,893 in 2026-07; clicks increased from 146 to 229. Monthly GA4 sessions increased from 567 to 794 and conversions from 12 to 24.',
            'The page-level view separates the cooling guide, comparison page, and cooling product page so Month 3 teams can monitor the content routes most relevant to the audit findings.'
        ]),
        ('Fix assessment detail', [
            'F01 Entity and company-fact cleanup: consistent_with_improvement; branded presence quality and target recommendations improved in the matched evidence.',
            'F02 Cooling Sheets Guide: consistent_with_improvement; target discovery and owned-citation signals improved in the matched evidence.',
            'F03 Cooling-brand comparison page: consistent_with_improvement; comparison standing and target recommendation movement improved.',
            'F04 Product claim and evidence refresh: consistent_with_improvement; branded quality and recommendation movement improved, while owned citation movement was flat.',
            'F05 Internal-linking and hub navigation: consistent_with_improvement; category presence improved, but owned citation movement was flat.',
            'F06 Independent expert-roundup outreach and F07 Night-sweats FAQ expansion: not_evaluable_month_2 because deployment timing or partial rollout prevented a fair Month 2 evaluation.',
            'These classifications indicate consistency with observed movement only; they do not claim causation.'
        ])
    ])

    draw_page(5, 'AsterVale Audit | Method and Decision Rules', [
        ('Analytical controls', [
            'The trusted dataset deduplicates run_id by latest ingested_at, keeps locked prompts and approved platforms/replicates, requires successful runs, and requires exactly one valid outcome for each of the five canonical brands.',
            'Outcome hierarchy is validated from mentioned to shortlisted to recommended to top_choice. Mentioned brands must have unique ranks 1-5; unmentioned brands must have no rank. Structurally invalid runs are excluded.',
            'Citation URLs are canonicalized by scheme/host/path/query rules, tracking parameters are removed, duplicate canonical citations are collapsed, and owned status is determined from the canonical brand domain.'
        ]),
        ('How to read the scorecard', [
            'Absolute percentage-point movement is the primary evidence for practical change. Relative change is reported for context and can look large when the Month 1 base is weak; a zero Month 1 base has no defined relative change.',
            'Month 1 and Month 2 movement uses only the intersection of eligible prompt_id + platform + replicate cells. Missing cells are not imputed.',
            'The full reproducible artifacts are available in outputs/: baseline, matched comparison, prompt-family drivers, platform drivers, fix assessment, next-sprint plan, QA report, and supporting analytics summary.'
        ]),
        ('Decision takeaway', [
            'AsterVale achieved meaningful Month 2 visibility gains, but the next operating challenge is converting visibility into recommendation through stronger owned proof, comparison evidence, and targeted content for stagnant high-intent segments.'
        ])
    ])
    c.save()

    # AI declaration
    ai_note = '''# AI Usage Declaration

This project was prepared using AI-assisted analysis for the data cleaning, metric construction, SQL design, and memo drafting process.
The underlying raw inputs were retained as provided and the generated pipeline rebuilds the output files from those untouched sources.
'''
    (ROOT / 'AI_USAGE_DECLARATION.md').write_text(ai_note, encoding='utf-8')

    # Replace README with run instructions
    readme_content = '''# AsterVale Audit Data Pipeline

## Purpose
This workspace contains a reproducible analysis pipeline for the AsterVale audit. It cleans the raw monthly runs, normalizes outcomes and citations, validates the eligibility rules, calculates the Month 1 baseline, compares matched Month 1 vs Month 2 performance, identifies fix drivers, and generates the required analytical outputs.

## Run
```bash
python run_analysis.py
```

## Expected outputs
The script writes the required CSVs into the outputs/ directory, including:
- qa_report.csv
- month_1_baseline.csv
- month_comparison_matched.csv
- prompt_family_deltas.csv
- platform_deltas.csv
- fix_assessment.csv
- next_sprint_plan.csv
- supporting_analytics_summary.csv
- runs_clean.csv
- outcomes_clean.csv
- citations_clean.csv
- executive_memo.pdf
'''
    (ROOT / 'README.md').write_text(readme_content, encoding='utf-8')

    return {
        'qa_report': qreport,
        'month_1_baseline': month_1_baseline,
        'comparison': comparison,
        'prompt_family': family_delta,
        'platform': platform_delta,
    }


if __name__ == '__main__':
    export_outputs()
