import re
import pandas as pd
import json
import os


"""
Margin - circumscribed, macrolobulated, microlobulated, indistinct, angular, spiculated
Shape - oval, round, Irregular
Orientation - parallel, not parallel
Echo pattern - anechoic, hypoechoic, isoechoic, hyperechoic, complex, heterogeneous
Posterior Features - No posterior features, enhancement, shadowing, combined pattern
Lesion Boundary - abrupt interface, echogenic halo, architectural distortion
(ignore multi lesion cases)
"""


def extract_ultrasound_findings(text):
    """
    Extract ultrasound section from radiology findings text.

    If text contains both MAMMO and ULTRASOUND sections, extracts only the
    ULTRASOUND section. Otherwise returns the full text.

    Args:
        text: Raw radiology findings text

    Returns:
        str: Extracted ultrasound findings text, or None if input is null
    """
    if pd.isna(text):
        return None

    # Convert to uppercase for section search
    text_upper = text.upper()

    # Look for ULTRASOUND or SONOGRAPHIC section
    ultrasound_match = re.search(r'\b(ULTRASOUND|SONO)\b', text_upper)

    # Find all MAMMO matches
    mammo_matches = list(re.finditer(r'\bMAMMO', text_upper))
    
    if not mammo_matches:
        # return full text
        return text
    
    # Filter out MAMMO matches in "due for mammo" or "correlat" context
    valid_mammo_found = False
    for mammo_match in mammo_matches:
        # Look back from mammo to last period (or start of text)
        lookback_start = text_upper.rfind('.', 0, mammo_match.start())
        if lookback_start == -1:
            lookback_start = 0

        lookback_text = text_upper[lookback_start:mammo_match.start()]

        # Check if "DUE" or "CORRELAT" appears between period and MAMMO
        if 'DUE' not in lookback_text and 'CORRELAT' not in lookback_text:
            # Valid MAMMO section found
            valid_mammo_found = True
            break

    # Only extract ULTRASOUND section if valid MAMMO section exists AND ultrasound section found
    if ultrasound_match:
        # Extract text after ULTRASOUND keyword
        section_start = ultrasound_match.start()
        return text[section_start:].strip()
    elif valid_mammo_found:
        # MAMMO exists but no ULTRASOUND section - this is NOT an ultrasound report
        return None  # Or return empty string to skip this record
    else:
        # No MAMMO section found - assume entire text is ultrasound
        return text


class UltrasoundNegationParser:
    """Custom parser for ultrasound findings with radiology-specific negation"""
    
    def __init__(self):
        # Margin priority: circumscribed is lowest priority (1), others are higher (2)
        # If any higher priority margin exists, it overwrites circumscribed
        self.margin_priority = {
            'circumscribed': 1,
            'macrolobulated': 2,
            'microlobulated': 2,
            'indistinct': 2,
            'angular': 2,
            'spiculated': 2,
        }
        
        # Echo pattern priority and types
        # Priority: complex (3) > heterogeneous (2) > basic patterns (1)
        self.echo_priority = {
            'anechoic': 1,
            'hypoechoic': 1,
            'isoechoic': 1,
            'hyperechoic': 1,
            'heterogeneous': 2,
            'complex': 3,
        }
        
        # Basic echo patterns that should be consolidated to heterogeneous if multiple found
        self.basic_echo_patterns = {'anechoic', 'hypoechoic', 'isoechoic', 'hyperechoic'}
        
        # Pre-negation patterns (appear BEFORE the concept)
        self.pre_negation_patterns = [
            r'\bno\b',
            r'\bwithout\b',
            r'\babsent\b',
            r'\bdenies\b',
            r'\bnot\s+(?:seen|present|identified|appreciated|visible)\b',
            r'\bfree\s+of\b',
            r'\backs?\b',
            r'\bnegative\s+for\b',
            r'\bno\s+evidence\s+of\b',
            r'\bno\s+longer\s+(?:seen|visible|present|identified)\b',
        ]
        
        # Post-negation patterns (appear AFTER the concept)
        self.post_negation_patterns = [
            r'\b(?:is|are|was|were)\s+no\s+longer\s+(?:seen|visible|present|identified|appreciated)\b',
            r'\b(?:is|are|was|were)\s+not\s+(?:seen|visible|present|identified|appreciated)\b',
            r'\b(?:has|have)\s+resolved\b',
            r'\b(?:is|are)\s+absent\b',
            r'\b(?:was|were)\s+ruled\s+out\b',
        ]
        
        # Terminators that end negation scope
        self.terminators = [
            r'\.',
            r'\bbut\b',
            r'\bhowever\b',
            r'\balthough\b',
            r'\bshowing\b',
            r'\bdemonstrat(?:es|ing)\b',
            r'\bpresent\b',  # unless it's "not present"
        ]
        
        # Feature patterns
        self.feature_patterns = {
            'margin': [
                (r'\bcircumscribed\b', 'circumscribed'),
                (r'\b(?:macro|macrolobulated)\b', 'macrolobulated'),
                (r'\b(?:micro|microlobulated)\b', 'microlobulated'),
                (r'\bindistinct\b', 'indistinct'),
                (r'\bangular\b', 'angular'),
                (r'\bspiculat(?:ed|ion)\b', 'spiculated'),
            ],
            'shape': [
                (r'\b(?:oval|ovoid)\b', 'oval'),
                (r'\bround\b', 'round'),
                (r'\birregular\b', 'irregular'),
            ],
            'orientation': [
                (r'\bparallel\b', 'parallel'),
                (r'\bnot\s+parallel\b', 'not parallel'),
                (r'\bantiparallel\b', 'not parallel'),
            ],
            'echo': [
                (r'\banecho(?:ic|genicity)\b', 'anechoic'),
                (r'\bhypoecho(?:ic|genicity)\b', 'hypoechoic'),
                (r'\bisoecho(?:ic|genicity)\b', 'isoechoic'),
                (r'\bhyperecho(?:ic|genicity)\b', 'hyperechoic'),
                (r'\bcomplex\b', 'complex'),
                (r'\bheterogene?ous\b', 'heterogeneous'),
            ],
            'posterior': [
                (r'\benhancement\b', 'enhancement'),
                (r'\bshadowing\b', 'shadowing'),
                (r'\bno\s+posterior\s+(?:acoustic\s+)?features?\b', 'no posterior features'),
            ],
            'boundary': [
                (r'\babrupt\s+interface\b', 'abrupt interface'),
                (r'\bechogenic\s+halo\b', 'echogenic halo'),
                (r'\barchitectural\s+distortion\b', 'architectural distortion'),
            ],
        }
    
    def normalize_text(self, text):
        """
        Normalize the text by extracting only the ultrasound section if it exists
        and if "MAMMO" appears before "ULTRASOUND".
        Returns the normalized text and normalization info for the audit.
        """
        if pd.isna(text):
            return None, {
                'original_length': 0,
                'normalized_length': 0,
                'mammo_found': False,
                'ultrasound_section_found': False,
                'normalization_applied': False
            }
        
        # Store original text length for the audit
        original_length = len(text)
        
        # Convert to uppercase temporarily for the section search
        text_upper = text.upper()
        
        # Look for ULTRASOUND (no colon required)
        ultrasound_match = re.search(r'\bULTRASOUND\b', text_upper)
        
        normalization_info = {
            'original_length': original_length,
            'mammo_found': False,
            'ultrasound_section_found': False,
            'normalization_applied': False
        }
        
        # Check if we should apply section extraction
        should_extract_section = False
        
        if ultrasound_match:
            normalization_info['ultrasound_section_found'] = True
            
            # Find all MAMMO matches
            mammo_matches = list(re.finditer(r'\bMAMMO', text_upper))
            
            # Filter out MAMMO matches that are in "due for mammo" or "correlat" context
            valid_mammo_found = False
            for mammo_match in mammo_matches:
                # Look back from mammo to last period (or start of text)
                lookback_start = text_upper.rfind('.', 0, mammo_match.start())
                if lookback_start == -1:
                    lookback_start = 0
                
                lookback_text = text_upper[lookback_start:mammo_match.start()]
                
                # Check if "DUE" or "CORRELAT" appears between the period and MAMMO
                if 'DUE' not in lookback_text and 'CORRELAT' not in lookback_text:
                    # This is a valid MAMMO (not in "due for mammo" or "correlates with mammo" context)
                    valid_mammo_found = True
                    normalization_info['mammo_found'] = True
                    normalization_info['mammo_position'] = mammo_match.start()
                    break
            
            if valid_mammo_found:
                should_extract_section = True
        
        # Only extract the ULTRASOUND section if we found both MAMMO and ULTRASOUND
        if should_extract_section:
            # Extract text after first appearance of ULTRASOUND
            section_start = ultrasound_match.start()
            normalized_text = text[section_start:].strip()
            
            normalization_info['normalization_applied'] = True
            normalization_info['section_start_pos'] = section_start
            normalization_info['normalized_length'] = len(normalized_text)
        else:
            # No ULTRASOUND found or no valid MAMMO, use the entire text
            normalized_text = text
            normalization_info['normalized_length'] = original_length
        
        return normalized_text, normalization_info
        
    def is_in_axillary_context(self, text, match_start, match_end):
        """Check if match is related to axillary nodes"""
        window = 100
        context_start = max(0, match_start - window)
        context_end = min(len(text), match_end + window)
        context = text[context_start:context_end].lower()
        
        axillary_indicators = [
            'axillary', 'axilla', 'lymph node', 'lymph nodes',
        ]
        
        return any(indicator in context for indicator in axillary_indicators)
    
    def is_negated(self, text, match_start, match_end):
        """Check if a feature at given position is negated"""
        text_lower = text.lower()
        
        # Look back up to 150 characters
        lookback_text = text_lower[max(0, match_start - 150):match_start]
        
        # Look ahead up to 150 characters (for post-negation)
        lookahead_text = text_lower[match_end:min(len(text_lower), match_end + 150)]
        
        # Check for terminators in lookback (which end negation scope)
        # Find the LAST (rightmost) terminator, not the first
        last_terminator_pos = -1
        for terminator in self.terminators:
            for match in re.finditer(terminator, lookback_text):
                last_terminator_pos = max(last_terminator_pos, match.end())
        
        if last_terminator_pos >= 0:
            # Only consider text after the last terminator
            lookback_text = lookback_text[last_terminator_pos:]
        
        # Check for PRE-negation patterns (before the match)
        for negation in self.pre_negation_patterns:
            if re.search(negation, lookback_text):
                return True
        
        # Check for POST-negation patterns (after the match)
        for negation in self.post_negation_patterns:
            if re.search(negation, lookahead_text):
                return True
        
        return False
    
    def apply_priority_rules(self, feature_type, feature_values):
        """
        Generic priority application for any feature type.
        Uses priority dictionaries and feature-specific consolidation rules.
        
        Args:
            feature_type: 'margin', 'echo', etc.
            feature_values: set of feature values
            
        Returns:
            Consolidated set of feature values after applying priority rules
        """
        if not feature_values or len(feature_values) <= 1:
            return feature_values
        
        # Get the priority map for this feature type
        priority_map = getattr(self, f'{feature_type}_priority', None)
        if not priority_map:
            return feature_values
        
        # Apply feature-specific rules
        if feature_type == 'margin':
            # Rule: Keep only highest priority items
            # If low priority (circumscribed) + high priority exist, remove low priority
            max_priority = max(priority_map.get(v, 0) for v in feature_values)
            if max_priority > 1:
                return {v for v in feature_values if priority_map.get(v, 0) == max_priority}
            return feature_values
        
        elif feature_type == 'echo':
            # Rule 1: Highest priority (complex) takes precedence over everything
            if 'complex' in feature_values:
                return {'complex'}
            
            # Rule 2: Mid priority (heterogeneous) takes precedence over basic patterns
            if 'heterogeneous' in feature_values:
                return {'heterogeneous'}
            
            # Rule 3: Multiple basic patterns consolidate to heterogeneous
            basic_patterns = feature_values & self.basic_echo_patterns
            if len(basic_patterns) >= 2:
                return {'heterogeneous'}
            
            return feature_values
        
        # Default: return as-is for unknown feature types
        return feature_values
    
    def parse_findings(self, findings_text):
        """Parse findings text and return both feature values and JSON audit"""
        if pd.isna(findings_text):
            return {
                'features': {f: None for f in self.feature_patterns.keys()},
                'audit': {
                    'original_text': None,
                    'normalization': None,
                    'decisions': []
                }
            }
        
        # Normalize text - extract ultrasound section if exists and MAMMO is before it
        normalized_text, normalization_info = self.normalize_text(findings_text)
        
        if normalized_text is None:
            return {
                'features': {f: None for f in self.feature_patterns.keys()},
                'audit': {
                    'original_text': findings_text,
                    'normalization': normalization_info,
                    'decisions': []
                }
            }
        
        text_lower = normalized_text.lower()
        decisions = []
        feature_values = {f: set() for f in self.feature_patterns.keys()}  # Changed to set()
        
        for feature_type, patterns in self.feature_patterns.items():
            for pattern, value_name in patterns:
                for match in re.finditer(pattern, text_lower):
                    # Skip if in axillary node context
                    #if self.is_in_axillary_context(text_lower, match.start(), match.end()):
                    #    continue
                    
                    is_neg = self.is_negated(text_lower, match.start(), match.end())
                    
                    # Get some context around the match
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text_lower), match.end() + 50)
                    context_text = text_lower[context_start:context_end]
                    
                    decision = {
                        'feature_type': feature_type,
                        'value': value_name,
                        'negated': is_neg,
                        'position': match.start(),
                        'match_text': match.group(0),
                        'context': context_text
                    }
                    
                    decisions.append(decision)
                    
                    # Only add to feature values if NOT negated
                    if not is_neg:
                        feature_values[feature_type].add(value_name)  # Changed to add()
        
        # Apply priority rules for margin and echo features
        for feature_type in ['margin', 'echo']:
            if feature_type in feature_values and feature_values[feature_type]:
                original_values = feature_values[feature_type].copy()
                feature_values[feature_type] = self.apply_priority_rules(feature_type, feature_values[feature_type])
                
                # Add audit info about priority application
                if original_values != feature_values[feature_type]:
                    decisions.append({
                        'feature_type': feature_type,
                        'value': 'PRIORITY_APPLIED',
                        'negated': False,
                        'position': -1,
                        'match_text': f"Original: {original_values}, After priority: {feature_values[feature_type]}",
                        'context': f'{feature_type.capitalize()} values consolidated based on priority rules'
                    })
        
        # Sort decisions by position for readability
        decisions.sort(key=lambda x: x['position'])
        
        # Convert feature sets to comma-separated strings (only if values exist)
        features = {}
        for feature_type, values in feature_values.items():
            if values:
                features[feature_type] = ', '.join(sorted(values))  # Sort for consistency
            else:
                features[feature_type] = None
        
        return {
            'features': features,
            'audit': {
                'original_text': findings_text,
                'normalized_text': normalized_text,
                'normalization': normalization_info,
                'decisions': decisions
            }
        }

def add_ultrasound_classifications(radiology_df, output_path):
    """Add classifications using custom parser with audit saved as separate JSON file"""
    parser = UltrasoundNegationParser()

    # Filter for ultrasound modality (all lateralities, including BILATERAL)
    mask = radiology_df['MODALITY'] == 'US'
    filtered_df = radiology_df[mask].copy()

    print(f"Processing {len(filtered_df)} records with custom parser...")

    # Step 1: Extract ultrasound findings section and add to new column
    print("Extracting ultrasound findings sections...")
    if 'ultrasound_findings' not in radiology_df.columns:
        radiology_df['ultrasound_findings'] = None

    for idx in filtered_df.index:
        findings_text = radiology_df.loc[idx, 'FINDINGS']
        ultrasound_text = extract_ultrasound_findings(findings_text)
        radiology_df.loc[idx, 'ultrasound_findings'] = ultrasound_text

    # Step 2: Ensure all feature columns exist
    for feature_type in parser.feature_patterns.keys():
        if feature_type not in radiology_df.columns:
            radiology_df[feature_type] = None

    # Step 3: Parse extracted ultrasound findings
    audit_data = {}

    for idx, row in filtered_df.iterrows():
        # Use extracted ultrasound findings for parsing
        ultrasound_text = radiology_df.loc[idx, 'ultrasound_findings']
        result = parser.parse_findings(ultrasound_text)

        # Update feature columns with non-negated values only
        for feature_type, value in result['features'].items():
            radiology_df.loc[idx, feature_type] = value

        # Add to audit data - use a unique identifier from the row
        record_id = row.get('Study_ID', str(idx))
        audit_data[record_id] = result['audit']

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    audit_output_path = f'{output_path}/ultrasound_parsing_audit.json'
    # Save audit data to JSON file
    with open(audit_output_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2)

    print(f"Parsing audit saved to {os.path.abspath(audit_output_path)}")

    return radiology_df