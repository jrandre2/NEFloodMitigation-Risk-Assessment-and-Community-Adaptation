# -*- coding: utf-8 -*-
"""
Absentee Residential Owner Name Classification Script
==============================================================
Tests the effectiveness of owner name classification on absentee residential properties.
Excludes owner-occupied properties and commercial/industrial parcels.
Outputs detailed statistics and sample classifications for review.

This script:
1. Loads all parcels from the geodatabase
2. Identifies residential properties using land use codes
3. Excludes owner-occupied properties (where owner address matches property address)
4. Classifies remaining absentee owners as Individual, Corporation, LLC, Trust, etc.
5. Analyzes patterns by distance and entity type
6. Outputs visualizations and detailed reports
"""

import os
import re
import time
import arcpy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────
# Update these paths to match your setup
PROJECT_DIR = r"C:\Mac\Home\Documents\ArcGIS\Projects\OwnerDistanceProject"
GDB = os.path.join(PROJECT_DIR, "OwnerDistanceProject.gdb")
PARCELS_FC = os.path.join(GDB, "Parcels")

# Output directory for results
OUTPUT_DIR = os.path.join(PROJECT_DIR, "owner_classification_results")

# Fields to check for owner names (in priority order)
OWNER_NAME_FIELDS = ["Current_Ow", "OW1_Name", "Owner_Name", "Owner1", "Street_Add", "OW1_Addres"]

# Distance field for analysis
DISTANCE_FIELD = "OwnerDist_km"

# Land use field and codes for residential
LAND_USE_FIELD = "Property_P"
RESIDENTIAL_CODES = [1]  # Based on your mapping

# Address fields for checking owner-occupied status
PARCEL_ADDR_FIELDS = ["Ph_Full_Ad", "Situs_Addr", "Ph_Full_Add", "Property_Address", "Site_Address"]
OWNER_ADDR_FIELDS = ["Street_Add", "OW1_Addres", "Current_Ow", "Mail_Add", "Owner_Add", "Mailing_Address", "Owner_Address"]
PARCEL_ZIP = "ParcelZIP5"
OWNER_ZIP_FIELDS = ["Zip", "OwnerZIP5", "Mail_ZIP", "Owner_ZIP"]

# Owner category field (if available)
OWNER_CATEGORY = "OwnerCategory"

# ── Helper Functions for Identifying Absentee Owners ──────────────────
def standardize_address(addr: str) -> str:
    """Standardize address for comparison."""
    if pd.isna(addr) or addr == "" or addr is None:
        return ""
    
    addr = str(addr).upper().strip()
    
    # Remove PO BOX addresses
    if "PO BOX" in addr or "P.O. BOX" in addr or "P O BOX" in addr:
        return ""
    
    # Remove punctuation and normalize
    addr = addr.replace(".", "").replace(",", "").replace("-", " ").replace("'", "")
    addr = addr.replace("  ", " ").strip()
    
    # Common replacements - more comprehensive
    replacements = {
        " STREET": " ST", " AVENUE": " AVE", " ROAD": " RD", " DRIVE": " DR",
        " LANE": " LN", " BOULEVARD": " BLVD", " PLACE": " PL", " COURT": " CT",
        " CIRCLE": " CIR", " TRAIL": " TRL", " PARKWAY": " PKWY", " HIGHWAY": " HWY",
        " SQUARE": " SQ", " TERRACE": " TER", " PLAZA": " PLZ", " ALLEY": " ALY",
        " NORTH ": " N ", " SOUTH ": " S ", " EAST ": " E ", " WEST ": " W ",
        " NORTHEAST ": " NE ", " NORTHWEST ": " NW ", " SOUTHEAST ": " SE ", " SOUTHWEST ": " SW ",
        " FIRST ": " 1ST ", " SECOND ": " 2ND ", " THIRD ": " 3RD ", " FOURTH ": " 4TH ",
        " FIFTH ": " 5TH ", " SIXTH ": " 6TH ", " SEVENTH ": " 7TH ", " EIGHTH ": " 8TH ",
        " NINTH ": " 9TH ", " TENTH ": " 10TH ",
        "  ": " "
    }
    
    for old, new in replacements.items():
        addr = addr.replace(old, new)
    
    # Remove apartment/unit numbers more aggressively
    addr = re.sub(r'\s+(APT|APARTMENT|UNIT|SUITE|STE|#|BLDG|BUILDING|FL|FLOOR|RM|ROOM)\s*[A-Z0-9\-]*', '', addr)
    addr = re.sub(r'\s+#[A-Z0-9\-]*', '', addr)
    
    return " ".join(addr.split())

def addresses_match(addr1: str, addr2: str, zip1: Any = None, zip2: Any = None, debug: bool = False) -> bool:
    """Check if two addresses match with improved matching logic."""
    # Convert to string and handle nulls
    addr1 = str(addr1) if pd.notna(addr1) else ""
    addr2 = str(addr2) if pd.notna(addr2) else ""
    
    # Standardize addresses
    std_addr1 = standardize_address(addr1)
    std_addr2 = standardize_address(addr2)
    
    if debug:
        print(f"Original 1: {addr1}")
        print(f"Original 2: {addr2}")
        print(f"Standard 1: {std_addr1}")
        print(f"Standard 2: {std_addr2}")
    
    if not std_addr1 or not std_addr2:
        return False
    
    # Check exact match
    if std_addr1 == std_addr2:
        return True
    
    # Extract components for fuzzy matching
    # Pattern: number + street name
    pattern = r'^(\d+)\s+(.+?)(?:\s+(?:ST|AVE|RD|DR|LN|BLVD|PL|CT|CIR|TRL|PKWY|HWY|SQ|TER|PLZ|ALY))?

def identify_absentee_residential(df: pd.DataFrame, 
                                   parcel_addr_fields: List[str],
                                   owner_addr_fields: List[str],
                                   land_use_field: str,
                                   residential_codes: List[int]) -> pd.DataFrame:
    """Identify absentee residential properties."""
    print("\nIdentifying absentee residential properties...")
    
    # First identify residential properties
    if land_use_field in df.columns:
        df[land_use_field] = pd.to_numeric(df[land_use_field], errors='coerce')
        df['IsResidential'] = df[land_use_field].isin(residential_codes)
        residential_count = df['IsResidential'].sum()
        print(f"  Found {residential_count:,} residential properties")
    else:
        print(f"  Warning: Land use field '{land_use_field}' not found - analyzing all properties")
        df['IsResidential'] = True
        residential_count = len(df)
    
    # Find best address fields to use
    parcel_addr_field = None
    for field in parcel_addr_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            if non_null > 0:
                parcel_addr_field = field
                print(f"  Using parcel address field: {parcel_addr_field} ({non_null:,} non-null values)")
                break
    
    owner_addr_field = None
    for field in owner_addr_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            if non_null > 0:
                owner_addr_field = field
                print(f"  Using owner address field: {owner_addr_field} ({non_null:,} non-null values)")
                break
    
    # Find ZIP fields
    parcel_zip_field = PARCEL_ZIP if PARCEL_ZIP in df.columns else None
    owner_zip_field = None
    for field in OWNER_ZIP_FIELDS:
        if field in df.columns:
            owner_zip_field = field
            break
    
    # Check for owner-occupied
    if parcel_addr_field and owner_addr_field:
        print(f"  Checking address matching...")
        
        # Initialize match column
        df['AddressMatch'] = False
        
        # Process in chunks for better performance and memory usage
        chunk_size = 10000
        total_residential = df['IsResidential'].sum()
        matched_count = 0
        
        # First, let's check a sample to debug
        print("\n  Checking sample addresses for debugging...")
        sample_df = df[df['IsResidential']].head(10)
        sample_matches = 0
        
        for idx, row in sample_df.iterrows():
            p_addr = row.get(parcel_addr_field, "")
            o_addr = row.get(owner_addr_field, "")
            p_zip = row.get(parcel_zip_field) if parcel_zip_field else None
            o_zip = row.get(owner_zip_field) if owner_zip_field else None
            
            match = addresses_match(p_addr, o_addr, p_zip, o_zip, debug=(idx == sample_df.index[0]))
            if match:
                sample_matches += 1
        
        print(f"  Sample match rate: {sample_matches}/10 ({sample_matches*10:.0f}%)")
        print()
        
        # Now process all records
        processed = 0
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk_indices = df.index[i:chunk_end]
            chunk_df = df.loc[chunk_indices]
            
            # Get residential properties in this chunk
            res_mask = chunk_df['IsResidential']
            res_indices = chunk_indices[res_mask]
            
            if len(res_indices) > 0:
                # Process each residential property
                for idx in res_indices:
                    p_addr = df.loc[idx, parcel_addr_field]
                    o_addr = df.loc[idx, owner_addr_field]
                    p_zip = df.loc[idx, parcel_zip_field] if parcel_zip_field else None
                    o_zip = df.loc[idx, owner_zip_field] if owner_zip_field else None
                    
                    # Convert to string and check
                    p_addr_str = str(p_addr) if pd.notna(p_addr) else ""
                    o_addr_str = str(o_addr) if pd.notna(o_addr) else ""
                    
                    if p_addr_str and o_addr_str:  # Both addresses must be non-empty
                        if addresses_match(p_addr_str, o_addr_str, p_zip, o_zip):
                            df.loc[idx, 'AddressMatch'] = True
                            matched_count += 1
                
                processed += len(res_indices)
            
            if processed % 50000 == 0 and processed > 0:
                print(f"    Processed {processed:,} residential records... ({matched_count:,} matches so far)")
        
        df['IsOwnerOccupied'] = df['IsResidential'] & df['AddressMatch']
        owner_occupied_count = df['IsOwnerOccupied'].sum()
        print(f"\n  Found {owner_occupied_count:,} owner-occupied residential properties")
        
        # Debug info if match rate seems low
        if residential_count > 0:
            match_rate = owner_occupied_count / residential_count * 100
            if match_rate < 30 and residential_count > 10000:
                print("\n  Warning: Low owner-occupied rate detected. Checking more samples...")
                
                # Show some examples of non-matching addresses
                non_match_sample = df[(df['IsResidential']) & (~df['AddressMatch'])].head(5)
                print("\n  Sample non-matching addresses:")
                for idx, row in non_match_sample.iterrows():
                    p_addr = row.get(parcel_addr_field, "")
                    o_addr = row.get(owner_addr_field, "")
                    print(f"    Parcel: {p_addr[:50]}")
                    print(f"    Owner:  {o_addr[:50]}")
                    print(f"    Standardized P: {standardize_address(str(p_addr))[:50]}")
                    print(f"    Standardized O: {standardize_address(str(o_addr))[:50]}")
                    print()
    else:
        print("  Warning: Address fields not found - cannot identify owner-occupied")
        df['IsOwnerOccupied'] = False
        owner_occupied_count = 0
    
    # Identify absentee residential
    df['IsAbsenteeResidential'] = df['IsResidential'] & ~df['IsOwnerOccupied']
    absentee_count = df['IsAbsenteeResidential'].sum()
    
    print(f"\n  Summary:")
    print(f"    Total properties: {len(df):,}")
    print(f"    Residential: {residential_count:,}")
    print(f"    Owner-occupied: {owner_occupied_count:,}")
    print(f"    Absentee residential: {absentee_count:,}")
    
    if residential_count > 0:
        owner_occupied_rate = owner_occupied_count / residential_count * 100
        absentee_rate = absentee_count / residential_count * 100
        print(f"    Owner-occupied rate: {owner_occupied_rate:.1f}%")
        print(f"    Absentee rate: {absentee_rate:.1f}%")
    
    return df

# ── Enhanced Owner Name Classifier ─────────────────────────────────────
class EnhancedOwnerNameClassifier:
    """Enhanced classifier with improved pattern matching for residential property owners."""
    
    def __init__(self):
        # Expanded corporate indicators
        self.corp_suffixes = [
            r'\bINC\.?(?:\s|$)', r'\bINCORPORATED\b', r'\bCORP\.?(?:\s|$)', r'\bCORPORATION\b',
            r'\bCOMPANY\b', r'\bCO\.(?:\s|$)', r'\bLTD\.?(?:\s|$)', r'\bLIMITED\b',
            r'\bPLC\b', r'\bSA\b', r'\bAG\b', r'\bGMBH\b', r'\bNV\b', r'\bBV\b',
            r'\bCORPS?\b', r'\bINCORP\b', r'\bCORPORATED\b', r'\bCOMPANIES\b',
            r'\bCORPORAT(?:$|[^E])'  # For CORPORAT (truncated)
        ]
        
        # Enhanced LLC patterns
        self.llc_patterns = [
            r'\bLLC\.?(?:\s|$)', r'\bL\.L\.C\.(?:\s|$)', r'\bLLP\.?(?:\s|$)', 
            r'\bL\.L\.P\.(?:\s|$)', r'\bLP\.?(?:\s|$)', r'\bL\.P\.(?:\s|$)',
            r'\bLIMITED LIABILITY', r'\bLIMITED PARTNERSHIP', r'\bLTD LIABILITY\b',
            r'\bLIMITED LIAB\b', r'\bL L C\b', r'\bL C\b', r'\bLMT PART\b',
            r'\bLTD PARTNERSHIP\b', r'\bLIMITED PARTNERS\b'
        ]
        
        # Strong trust indicators
        self.strong_trust_indicators = [
            r'\bTR\s+(?:DTD|DATED)',  # TR DATED
            r'\bREVOCABLE.*TRUST',
            r'\bLIVING.*TRUST',
            r'\bFAMILY.*TRUST',
            r'\b(?:TRUSTEE|TRUSTEES)\s+(?:OF|FOR)\b',
        ]
        
        # Expanded trust patterns
        self.trust_patterns = [
            r'\bTRUST\b', r'\bTRUSTEE[S]?\b', r'\bREVOCABLE\b', r'\bIRREVOCABLE\b',
            r'\bLIVING TRUST\b', r'\bFAMILY TRUST\b', r'\b(?:TR|TRS|TRST)\b',
            r'\bTRST\b', r'\bREV TR\b', r'\bFAM TR\b', r'\bQTIP\b', r'\bCHARITABLE\b',
            r'\bTESTAMENTARY\b', r'\bGRANTOR\b', r'\bSURVIVOR[S]? TRUST\b',
            r'\bREV TRUST\b', r'\bREVOC TR\b', r'\bIRREV TR\b', r'\bU/A\b', r'\bU/T\b',
            r'\bUNDER AGREEMENT\b', r'\bUNDER TRUST\b', r'\b(?:DTD|DATED)\b',
            r'\bTRU\b', r'\bLIV TR\b', r'\bLIVING TR\b'
        ]
        
        # Estate patterns
        self.estate_patterns = [
            r'\bESTATE\b', r'\bEST\.(?:\s|$)', r'\bESTATE OF\b', r'\bDECEASED\b',
            r'\bEXECUTOR\b', r'\bADMINISTRATOR\b', r'\bHEIRS\b', r'\bSURVIVOR\b',
            r'\bESTATES\b', r'\bEST OF\b', r'\bHEIRS OF\b', r'\bSURVIVING\b',
            r'\bPERSONAL REP\b', r'\bPERS REP\b', r'\bEXEC\b', r'\bADMIN\b'
        ]
        
        # Government/Public patterns
        self.gov_patterns = [
            r'\bCITY OF\b', r'\bCOUNTY OF\b', r'\bSTATE OF\b', r'\bUNITED STATES\b',
            r'\bDEPT\.?(?:\s|$)', r'\bDEPARTMENT\b', r'\bAUTHORITY\b', r'\bMUNICIPAL\b',
            r'\bSCHOOL DISTRICT\b', r'\bHOUSING AUTHORITY\b', r'\bPUBLIC\b',
            r'\bGOVERNMENT\b', r'\bFEDERAL\b', r'\bUSA\b', r'\bU\.S\.A\.(?:\s|$)',
            r'\bUNIVERSITY\b', r'\bCOLLEGE\b', r'\bBOARD OF\b', r'\bCOMMISSION\b',
            r'\bAGENCY\b', r'\bDISTRICT\b', r'\bTOWNSHIP\b', r'\bVILLAGE OF\b',
            r'\bHOUSING AUTH\b', r'\bSCH DIST\b', r'\bSCHOOL DIST\b', r'\bSCHL DIST\b',
            r'\bCNTY\b', r'\bCOUNTY\b', r'\bFIRE DIST\b', r'\bFIRE RESCUE\b',
            r'\bSECRETARY\b.*\bHOUSING\b', r'\bHUD\b', r'\bREGENTS\b', r'\bBOARD\b.*\bREGENTS\b',
            r'\bPONCA TRIBE\b', r'\bTRIBE OF\b'
        ]
        
        # Religious/Non-profit patterns - expanded for residential area churches
        self.religious_nonprofit_patterns = [
            r'\bCHURCH\b', r'\bCHURCHES\b', r'\bCATHOLIC\b', r'\bBAPTIST\b', 
            r'\bLUTHERAN\b', r'\bMETHODIST\b', r'\bPRESBYTERIAN\b', r'\bEPISCOPAL\b',
            r'\bCHRISTIAN\b', r'\bCHRIST\b', r'\bJESUS\b', r'\bGOSPEL\b', 
            r'\bMINISTRY\b', r'\bMINISTRIES\b', r'\bTEMPLE\b', r'\bSYNAGOGUE\b', 
            r'\bMOSQUE\b', r'\bPARISH\b', r'\bCONGREGATION\b', r'\bJEHOVAH\b', 
            r'\bSAINT[S]?\b', r'\bST\.?\s', r'\bCHAPEL\b', r'\bFAITH\b',
            r'\bIGLESIA\b', r'\bCENTRO CRISTIANO\b', r'\bMINISTERIO\b',  # Spanish
            r'\bCHABAD\b', r'\bLUBAVITCH\b',  # Jewish
            r'\bPENTECOSTE[S]?\b', r'\bASSEMBLY OF GOD\b',
            r'\bHABITAT\s+FOR\s+HUMANITY\b', r'\bYMCA\b', r'\bYWCA\b', r'\bFOUNDATION\b',
            r'\bINSTITUTE\b', r'\bCHARITY\b', r'\bCHARITABLE\b', r'\bNONPROFIT\b', 
            r'\bNON-PROFIT\b', r'\bSOCIETY\b', r'\bASSOCIATION\b', r'\bLEAGUE\b',
            r'\bCOMMUNITY\s+CENTER\b', r'\bSALVATION ARMY\b', r'\bMISSION\b',
            r'\bRESCUE MISSION\b', r'\bHOPE MISSION\b'
        ]
        
        # Healthcare patterns
        self.healthcare_patterns = [
            r'\bHOSPITAL\b', r'\bMEDICAL\b', r'\bHEALTH\b', r'\bCLINIC\b',
            r'\bHEALTHCARE\b', r'\bHEALTH CENTER\b', r'\bNURSING\b', 
            r'\bHOSPICE\b', r'\bREHAB\b', r'\bREHABILITATION\b',
            r'\bPHYSICIANS?\b', r'\bDIALYSIS\b', r'\bANIMAL HEALTH\b'
        ]
        
        # Banking/Financial patterns
        self.bank_patterns = [
            r'\bBANK\b', r'\bCREDIT UNION\b', r'\bSAVINGS\b', r'\bFINANCIAL\b',
            r'\bBANKING\b', r'\bFIRST NATIONAL\b', r'\bNATIONAL BANK\b',
            r'\bBANK OF\b', r'\bBANK & TRUST\b', r'\bBANK AND TRUST\b'
        ]
        
        # Condominium/HOA patterns
        self.condo_patterns = [
            r'\bCONDO\b', r'\bCONDOMINIUM\b', r'\bCONDOMINION\b',
            r'\bCONDO ASSOC\b', r'\bCONDOMINIUM ASSOC\b', r'\bCONDOMINIUM ASSN\b',
            r'\bHOMEOWNERS?\b', r'\bHOA\b', r'\bTOWNHOME\b', r'\bTOWNHOUSE\b',
            r'\bOWNERS ASSOC\b', r'\bOWNERS ASSN\b'
        ]
        
        # Housing-specific patterns
        self.housing_patterns = [
            r'HOUSING AUTHOR',  # Housing Authority (truncated)
            r'SENIOR COTTAGES?',
            r'SENIOR HOUSING',
            r'AFFORDABLE HOUSING',
            r'GOOD SAMARITAN',
            r'COTTAGES?\b',
            r'SENIOR LIVING',
            r'ASSISTED LIVING',
            r'RETIREMENT'
        ]
        
        # Truncated organization patterns
        self.truncated_org_patterns = [
            r'HUMAN SERVIC$',  # Human Services
            r'NATURAL RESOUR$',  # Natural Resources
            r'COMM DEVELOP$',  # Community Development
            r'ASSOC?I?A?$',  # Association variants
            r'DEVELOP$',  # Development
            r'CORPORAT$',  # Corporation
            r'APARTMEN$',  # Apartments
            r'PROPERT$',  # Properties/Property
            r'COMMUNIT$',  # Community
            r'MANAGEMEN$',  # Management
            r'INVESTMEN$',  # Investment
            r'RESIDENTIA$',  # Residential
        ]
        
        # Business patterns - focused on residential property owners
        self.business_patterns = [
            r'\bPROPERTIES\b', r'\bHOLDINGS\b', r'\bGROUP\b', r'\bPARTNERS\b',
            r'\bVENTURES\b', r'\bENTERPRISES\b', r'\bDEVELOPMENT\b', r'\bMANAGEMENT\b',
            r'\bINVESTMENTS?\b', r'\bCAPITAL\b', r'\bREALTY\b', r'\bREAL ESTATE\b',
            r'\bASSET[S]?\b', r'\bEQUITY\b', r'\bFUND[S]?\b', r'\bPORTFOLIO\b',
            r'\bACQUISITIONS?\b', r'\bCONSTRUCTION\b', r'\bBUILDERS?\b',
            r'\bASSOCIATES\b', r'\bSERVICES\b', r'\bSOLUTIONS\b', r'\bCENTER\b',
            r'\bRENTALS?\b', r'\bLEASING\b', r'\bHOMES?\b', r'\bLAND\b', r'\bPROPERTY\b',
            r'\bESTATES\b', r'\bRESIDENTIAL\b', r'\bAPARTMENTS?\b', r'\bBROTHERS\b',
            r'\bFAMILY\b', r'\bPARTNERSHIP\b', r'\bJOINT VENTURE\b',
            r'\bMANOR\b', r'\bVILLAGE\b', r'\bCOMMONS\b', r'\bCROSSING\b',
            r'\bPARK\b', r'\bGARDENS?\b', r'\bPLACE\b', r'\bSQUARE\b',
            r'\bPOINTE?\b', r'\bRIDGE\b', r'\bCREEK\b', r'\bLAKE\b', r'\bHILLS?\b',
            r'\bVALLEY\b', r'\bVISTA\b', r'\bVIEW\b', r'\bHEIGHTS\b', r'\bMEADOWS?\b'
        ]
        
        # Personal name indicators - enhanced with ETA pattern
        self.personal_patterns = [
            r'\bJR\.?(?:\s|$)', r'\bSR\.?(?:\s|$)', r'\bIII\b', r'\bII\b', r'\bIV\b',
            r'\bMR\.?(?:\s|$)', r'\bMRS\.?(?:\s|$)', r'\bMS\.?(?:\s|$)', r'\bDR\.?(?:\s|$)',
            r'\bAND\b', r'\b&\b', r'\bOR\b', r'\bETAL\.?(?:\s|$)',
            r'\bET AL\.?(?:\s|$)', r'\bET UX\.?(?:\s|$)', r'\bET VIR\.?(?:\s|$)',
            r'\bETA\b',  # Common misspelling/truncation of ET AL
            r'\bHUSBAND\b', r'\bWIFE\b', r'\bSPOUSE\b', r'\bMARRIED\b',
            r'\bJOINT\b', r'\bTENANTS\b', r'\bJT TEN\b', r'\bJTWROS\b'
        ]
        
        # International name patterns
        self.international_patterns = {
            'indian': r'\b(?:REDDY|KUMAR|SINGH|PATEL|SHARMA|GUPTA|SHAH|RAO|NAIR|IYER)\b',
            'asian': r'\b(?:NGUYEN|TRAN|LE|PHAM|HOANG|HO|NGO|VU|DANG|BUI|DO|PHAN|DUONG|LAM)\b',
            'hispanic_special': r'\b(?:DE LA|DE LOS|DE LAS|DEL|DELA)\b',
        }
        
        # Common first names - pruned for likely property owners (older, predominantly male)
        self.common_first_names = set([
            # Traditional male names - more common among property owners
            'JOHN', 'DAVID', 'MICHAEL', 'ROBERT', 'JAMES', 'WILLIAM', 'RICHARD', 'THOMAS',
            'JOSEPH', 'CHARLES', 'CHRISTOPHER', 'DANIEL', 'MATTHEW', 'DONALD', 'KENNETH',
            'MARK', 'PAUL', 'STEVEN', 'ANDREW', 'GEORGE', 'EDWARD', 'RONALD', 'TIMOTHY',
            'GARY', 'LARRY', 'JEFFREY', 'FRANK', 'SCOTT', 'ERIC', 'STEPHEN', 'RAYMOND',
            'GREGORY', 'SAMUEL', 'BENJAMIN', 'PATRICK', 'JACK', 'DENNIS', 'JERRY', 'CARL',
            'HAROLD', 'DOUGLAS', 'HENRY', 'ARTHUR', 'PETER', 'ALBERT', 'WILLIE', 'WAYNE',
            'RALPH', 'EUGENE', 'RUSSELL', 'ROY', 'LOUIS', 'PHILIP', 'JOHNNY', 'ERNEST',
            'MARTIN', 'CRAIG', 'ALAN', 'CLARENCE', 'WALTER', 'CARL', 'BRUCE', 'KEITH',
            'NORMAN', 'MARVIN', 'VINCENT', 'GLENN', 'HOWARD', 'FRED', 'LEON', 'DALE',
            'CLIFFORD', 'CHESTER', 'LLOYD', 'MELVIN', 'VERNON', 'HERMAN', 'LEROY',
            'FRANCIS', 'STANLEY', 'LEONARD', 'NATHAN', 'JOEL', 'EDWIN', 'DON', 'GORDON',
            'DEAN', 'JEROME', 'JIM', 'TOM', 'BILL', 'MIKE', 'DAVE', 'RICK', 'BOB',
            'STEVE', 'JEFF', 'ROGER', 'TERRY', 'BRUCE', 'RANDY', 'BERNARD', 'NICHOLAS',
            'LEE', 'ALLEN', 'KEVIN', 'BRIAN', 'JOSHUA', 'RYAN', 'JACOB', 'JASON',
            'JUSTIN', 'BRANDON', 'JOSE', 'JUAN', 'MIGUEL', 'CARLOS', 'ANTONIO', 'MANUEL',
            'PEDRO', 'FRANCISCO', 'JESUS', 'LUIS', 'JORGE', 'ROBERTO', 'ALBERTO', 'FERNANDO',
            'PABLO', 'MARCO', 'RICARDO', 'EDUARDO', 'JAVIER', 'SERGIO', 'ARTURO', 'ALEJANDRO',
            'ENRIQUE', 'GERARDO', 'MARIO', 'ADRIAN', 'HECTOR', 'RAFAEL', 'OMAR', 'RAUL',
            'ARMANDO', 'ALFREDO', 'MARTIN', 'FELIX', 'ANDRES', 'JULIO', 'DIEGO', 'IVAN',
            'VICTOR', 'SAMUEL', 'GABRIEL', 'ALEXANDER', 'ANTHONY', 'MARCUS', 'MATTHEW',
            'LAWRENCE', 'WARREN', 'HERBERT', 'FREDERICK', 'CLIFTON', 'WALLACE', 'STANLEY',
            'EUGENE', 'RALPH', 'HOWARD', 'KENNETH', 'HAROLD', 'FRANCIS', 'BENJAMIN',
            'HARVEY', 'ARNOLD', 'GERALD', 'KARL', 'NEIL', 'STUART', 'MARSHALL', 'GUY',
            'GILBERT', 'ROLAND', 'THEODORE', 'BERNARD', 'EUGENE', 'HERBERT', 'NORMAN',
            'CLIFFORD', 'RAYMOND', 'WARREN', 'HOWARD', 'STANLEY', 'ARTHUR', 'ERNEST',
            'LEONARD', 'ALBERT', 'LAWRENCE', 'GORDON', 'NORMAN', 'RUSSELL', 'WALLACE',
            
            # Common female names - property owners (reduced set)
            'MARY', 'PATRICIA', 'LINDA', 'BARBARA', 'ELIZABETH', 'JENNIFER', 'MARIA',
            'SUSAN', 'MARGARET', 'DOROTHY', 'LISA', 'NANCY', 'KAREN', 'BETTY', 'HELEN',
            'SANDRA', 'DONNA', 'CAROL', 'RUTH', 'SHARON', 'MICHELLE', 'LAURA', 'SARAH',
            'KIMBERLY', 'DEBORAH', 'JESSICA', 'SHIRLEY', 'CYNTHIA', 'ANGELA', 'MELISSA',
            'BRENDA', 'ANNA', 'VIRGINIA', 'KATHLEEN', 'PAMELA', 'MARTHA', 'DEBRA',
            'AMANDA', 'STEPHANIE', 'CAROLYN', 'CHRISTINE', 'MARIE', 'JANET', 'CATHERINE',
            'FRANCES', 'CHRISTINA', 'JOAN', 'EVELYN', 'ALICE', 'JULIE', 'JOYCE', 'DIANE',
            'GLORIA', 'JEAN', 'TERESA', 'DORIS', 'JUDITH', 'ROSE', 'JANICE', 'KELLY',
            'NICOLE', 'JUDY', 'CHERYL', 'KATHY', 'THERESA', 'BEVERLY', 'DENISE', 'TAMMY',
            'IRENE', 'JANE', 'LORI', 'MARILYN', 'KATHRYN', 'LOUISE', 'SARA', 'ANNE',
            'JACQUELINE', 'WANDA', 'BONNIE', 'JULIA', 'RUBY', 'LOIS', 'TINA', 'PHYLLIS',
            'NORMA', 'PAULA', 'DIANA', 'ANNIE', 'LILLIAN', 'EMILY', 'ROBIN', 'PEGGY',
            'CRYSTAL', 'GLADYS', 'RITA', 'DAWN', 'CONNIE', 'FLORENCE', 'TRACY', 'EDNA',
            'TIFFANY', 'CARMEN', 'ROSA', 'CINDY', 'GRACE', 'WENDY', 'VICTORIA', 'EDITH',
            'KIM', 'SHERRY', 'SYLVIA', 'JOSEPHINE', 'THELMA', 'SHANNON', 'SHEILA', 'ETHEL',
            'ELLEN', 'ELAINE', 'MARJORIE', 'CARRIE', 'CHARLOTTE', 'MONICA', 'ESTHER',
            'PAULINE', 'EMMA', 'JUANITA', 'ANITA', 'RHONDA', 'HAZEL', 'AMBER', 'EVA',
            'APRIL', 'LESLIE', 'CLARA', 'LUCILLE', 'JAMIE', 'JOANNE', 'ELEANOR', 'VALERIE',
            'DANIELLE', 'MEGAN', 'ALICIA', 'SUZANNE', 'MICHELE', 'GAIL', 'BERTHA', 'DARLENE',
            'VERONICA', 'JILL', 'ERIN', 'GERALDINE', 'LAUREN', 'CATHY', 'JOANN', 'LORRAINE',
            'LYNN', 'SALLY', 'REGINA', 'ERICA', 'BEATRICE', 'DOLORES', 'BERNICE', 'AUDREY',
            'YVONNE', 'ANNETTE', 'SAMANTHA', 'MARION', 'DANA', 'STACY', 'ANA', 'RENEE',
            'IDA', 'VIVIAN', 'ROBERTA', 'HOLLY', 'BRITTANY', 'MELANIE', 'LORETTA', 'YOLANDA',
            'JEANETTE', 'LAURIE', 'KATIE', 'KRISTEN', 'VANESSA', 'ALMA', 'SUE', 'ELSIE',
            'BETH', 'MILDRED', 'ALLISON', 'TAMARA', 'SANDY', 'BELINDA', 'COLLEEN', 'BILLIE',
            
            # Additional unisex/shorter forms
            'JORDAN', 'TAYLOR', 'MORGAN', 'CASEY', 'JAMIE', 'ALEX', 'CHRIS', 'PAT',
            'SAM', 'ROBIN', 'TERRY', 'LEE', 'JEAN', 'DANA', 'LYNN', 'SHANNON', 'KIM',
            'TRACY', 'JESSE', 'CAMERON', 'BLAKE', 'DREW', 'RILEY'
        ])
        
        # Common last names - focused on established families
        self.common_last_names = set([
            'SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN', 'JONES', 'GARCIA', 'MILLER', 'DAVIS',
            'RODRIGUEZ', 'MARTINEZ', 'HERNANDEZ', 'LOPEZ', 'GONZALEZ', 'WILSON', 'ANDERSON',
            'THOMAS', 'TAYLOR', 'MOORE', 'JACKSON', 'MARTIN', 'LEE', 'PEREZ', 'THOMPSON',
            'WHITE', 'HARRIS', 'SANCHEZ', 'CLARK', 'RAMIREZ', 'LEWIS', 'ROBINSON', 'WALKER',
            'YOUNG', 'ALLEN', 'KING', 'WRIGHT', 'SCOTT', 'TORRES', 'NGUYEN', 'HILL', 'FLORES',
            'GREEN', 'ADAMS', 'NELSON', 'BAKER', 'HALL', 'RIVERA', 'CAMPBELL', 'MITCHELL',
            'CARTER', 'ROBERTS', 'GOMEZ', 'PHILLIPS', 'EVANS', 'TURNER', 'DIAZ', 'PARKER',
            'CRUZ', 'EDWARDS', 'COLLINS', 'REYES', 'STEWART', 'MORRIS', 'MORALES', 'MURPHY',
            'COOK', 'ROGERS', 'GUTIERREZ', 'ORTIZ', 'MORGAN', 'COOPER', 'PETERSON', 'BAILEY',
            'REED', 'KELLY', 'HOWARD', 'RAMOS', 'KIM', 'COX', 'WARD', 'RICHARDSON', 'WATSON',
            'BROOKS', 'CHAVEZ', 'WOOD', 'JAMES', 'BENNETT', 'GRAY', 'MENDOZA', 'RUIZ',
            'HUGHES', 'PRICE', 'ALVAREZ', 'CASTILLO', 'SANDERS', 'PATEL', 'MYERS', 'LONG',
            'ROSS', 'FOSTER', 'JIMENEZ', 'POWELL', 'JENKINS', 'PERRY', 'RUSSELL', 'SULLIVAN',
            'BELL', 'COLEMAN', 'BUTLER', 'HENDERSON', 'BARNES', 'GONZALES', 'FISHER', 'VASQUEZ',
            'SIMMONS', 'ROMERO', 'JORDAN', 'PATTERSON', 'ALEXANDER', 'HAMILTON', 'GRAHAM',
            'REYNOLDS', 'GRIFFIN', 'WALLACE', 'MORENO', 'WEST', 'COLE', 'HAYES', 'BRYANT',
            'HERRERA', 'GIBSON', 'ELLIS', 'TRAN', 'MEDINA', 'AGUILAR', 'STEVENS', 'MURRAY',
            'FORD', 'CASTRO', 'MARSHALL', 'OWENS', 'HARRISON', 'FERNANDEZ', 'MCDONALD',
            'WOODS', 'WASHINGTON', 'KENNEDY', 'WELLS', 'VARGAS', 'HENRY', 'CHEN', 'FREEMAN',
            'WEBB', 'TUCKER', 'GUZMAN', 'BURNS', 'CRAWFORD', 'OLSON', 'SIMPSON', 'PORTER',
            'HUNTER', 'GORDON', 'MENDEZ', 'SILVA', 'SHAW', 'SNYDER', 'MASON', 'DIXON',
            'MUNOZ', 'HUNT', 'HICKS', 'HOLMES', 'PALMER', 'WAGNER', 'BLACK', 'ROBERTSON',
            'BOYD', 'ROSE', 'STONE', 'SALAZAR', 'FOX', 'WARREN', 'MILLS', 'MEYER', 'RICE',
            'SCHMIDT', 'GARZA', 'DANIELS', 'FERGUSON', 'NICHOLS', 'STEPHENS', 'SOTO', 'WEAVER',
            'RYAN', 'GARDNER', 'PAYNE', 'GRANT', 'DUNN', 'KELLEY', 'SPENCER', 'HAWKINS',
            'ARNOLD', 'PIERCE', 'VAZQUEZ', 'HANSEN', 'PETERS', 'SANTOS', 'HART', 'BRADLEY',
            'KNIGHT', 'ELLIOTT', 'CUNNINGHAM', 'DUNCAN', 'ARMSTRONG', 'HUDSON', 'CARROLL',
            'LANE', 'RILEY', 'ANDREWS', 'ALVARADO', 'RAY', 'DELGADO', 'BERRY', 'PERKINS',
            'HOFFMAN', 'JOHNSTON', 'MATTHEWS', 'PENA', 'RICHARDS', 'CONTRERAS', 'WILLIS',
            'CARPENTER', 'LAWRENCE', 'SANDOVAL', 'GUERRERO', 'GEORGE', 'CHAPMAN', 'RIOS',
            'ESTRADA', 'ORTEGA', 'WATKINS', 'GREENE', 'NUNEZ', 'WHEELER', 'VALDEZ', 'HARPER',
            'BURKE', 'LARSON', 'SANTIAGO', 'MALDONADO', 'MORRISON', 'FRANKLIN', 'CARLSON',
            'AUSTIN', 'DOMINGUEZ', 'CARR', 'LAWSON', 'JACOBS', 'OBRIEN', 'LYNCH', 'SINGH',
            'VEGA', 'BISHOP', 'MONTGOMERY', 'OLIVER', 'JENSEN', 'HARVEY', 'WILLIAMSON',
            'GILBERT', 'DEAN', 'SIMS', 'ESPINOZA', 'HOWELL', 'LI', 'WONG', 'REID', 'HANSON',
            'LE', 'MCCOY', 'GARRETT', 'BURTON', 'FULLER', 'WANG', 'WEBER', 'WELCH', 'ROJAS',
            'LUCAS', 'MARQUEZ', 'FIELDS', 'PARK', 'YANG', 'LITTLE', 'BANKS', 'PADILLA', 'DAY',
            'WALSH', 'BOWMAN', 'SCHULTZ', 'LUNA', 'FOWLER', 'MEJIA', 'DAVIDSON', 'ACOSTA',
            'BREWER', 'MAY', 'HOLLAND', 'JUAREZ', 'NEWMAN', 'PEARSON', 'CURTIS', 'CORTEZ',
            'DOUGLAS', 'SCHNEIDER', 'JOSEPH', 'BARRETT', 'NAVARRO', 'FIGUEROA', 'KELLER',
            'AVILA', 'WADE', 'MOLINA', 'STANLEY', 'HOPKINS', 'CAMPOS', 'BARNETT', 'BATES',
            'CHAMBERS', 'CALDWELL', 'BECK', 'LAMBERT', 'MIRANDA', 'BYRD', 'CRAIG', 'AYALA',
            'LOWE', 'FRAZIER', 'POWERS', 'NEAL', 'LEONARD', 'GREGORY', 'CARRILLO', 'SUTTON',
            'FLEMING', 'RHODES', 'SHELTON', 'SCHWARTZ', 'NORRIS', 'JENNINGS', 'WATTS',
            'DURAN', 'WALTERS', 'COHEN', 'MCDANIEL', 'MORAN', 'PARKS', 'STEELE', 'VAUGHN',
            'BECKER', 'HOLT', 'DELEON', 'BARKER', 'TERRY', 'HALE', 'LEON', 'HAIL', 'BENSON',
            'HAYNES', 'HORTON', 'MILES', 'LYONS', 'PHAM', 'GRAVES', 'BUSH', 'THORNTON',
            'WOLFE', 'WARNER', 'CABRERA', 'MCKINNEY', 'MANN', 'ZIMMERMAN', 'DAWSON', 'LARA',
            'FLETCHER', 'PAGE', 'MCCARTHY', 'LOVE', 'ROBLES', 'CERVANTES', 'SOLIS', 'ERICKSON',
            'REEVES', 'CHANG', 'KLEIN', 'SALINAS', 'FUENTES', 'BALDWIN', 'DANIEL', 'SIMON',
            'VELASQUEZ', 'HARDY', 'HIGGINS', 'AGUIRRE', 'LIN', 'CUMMINGS', 'CHANDLER',
            'SHARP', 'BARBER', 'BOWEN', 'OCHOA', 'DENNIS', 'ROBBINS', 'LIU', 'RAMSEY',
            'FRANCIS', 'GRIFFITH', 'PAUL', 'BLAIR', 'OCONNOR', 'CARDENAS', 'PACHECO',
            'CROSS', 'CALDERON', 'QUINN', 'MOSS', 'SWANSON', 'CHAN', 'RIVAS', 'KHAN',
            'RODGERS', 'SERRANO', 'FITZGERALD', 'ROSALES', 'STEVENSON', 'CHRISTENSEN',
            'MANNING', 'GILL', 'CURRY', 'MCLAUGHLIN', 'HARMON', 'MCGEE', 'GROSS', 'DOYLE',
            'GARNER', 'NEWTON', 'BURGESS', 'REESE', 'WALTON', 'BLAKE', 'TRUJILLO', 'ADKINS',
            'BRADY', 'GOODMAN', 'ROMAN', 'WEBSTER', 'GOODWIN', 'FISCHER', 'HUANG', 'POTTER',
            'DELACRUZ', 'MONTOYA', 'TODD', 'WU', 'HINES', 'MULLINS', 'CASTANEDA', 'MALONE',
            'CANNON', 'TATE', 'MACK', 'SHERMAN', 'HUBBARD', 'HODGES', 'ZHANG', 'GUERRA',
            'WOLF', 'VALENCIA', 'FRANCO', 'SAUNDERS', 'ROWE', 'GALLAGHER', 'FARMER',
            'HAMMOND', 'HAMPTON', 'TOWNSEND', 'INGRAM', 'WISE', 'GALLEGOS', 'CLARKE',
            'BARTON', 'SCHROEDER', 'MAXWELL', 'WATERS', 'LOGAN', 'CAMACHO', 'STRICKLAND',
            'NORMAN', 'PERSON', 'COLON', 'PARSONS', 'FRANK', 'HARRINGTON', 'GLOVER',
            'OSBORNE', 'BUCHANAN', 'CASEY', 'FLOYD', 'PATTON', 'IBARRA', 'BALL', 'TYLER',
            'SUAREZ', 'BOWERS', 'OROZCO', 'SALAS', 'COBB', 'GIBBS', 'ANDRADE', 'BAUER',
            'CONNER', 'MOODY', 'ESCOBAR', 'MCGUIRE', 'LLOYD', 'MUELLER', 'HARTMAN',
            'FRENCH', 'KRAMER', 'MCBRIDE', 'POPE', 'LINDSEY', 'VELAZQUEZ', 'NORTON',
            'MCCORMICK', 'SPARKS', 'FLYNN', 'YATES', 'HOGAN', 'MARSH', 'MACIAS', 'VILLANUEVA',
            'ZAMORA', 'PRATT', 'STOKES', 'OWEN', 'BALLARD', 'LANG', 'BROCK', 'VILLARREAL',
            'CHARLES', 'DRAKE', 'BARRERA', 'CAIN', 'PATRICK', 'PINEDA', 'BURNETT', 'MERCADO',
            'SANTANA', 'SHEPHERD', 'BAUTISTA', 'ALI', 'SHAFFER', 'LAMB', 'TREVINO', 'MCKENZIE',
            'HESS', 'BEIL', 'OLSEN', 'COCHRAN', 'MORTON', 'NASH', 'WILKINS', 'PETERSEN',
            'BRIGGS', 'SHAH', 'ROTH', 'NICHOLSON', 'HOLLOWAY', 'LOZANO', 'FLOWERS',
            'RANGEL', 'HOOVER', 'ARIAS', 'SHORT', 'MORA', 'VALENZUELA', 'BRYAN', 'MEYERS',
            'WEISS', 'UNDERWOOD', 'BASS', 'GREER', 'SUMMERS', 'HOUSTON', 'CARSON', 'MORROW',
            'CLAYTON', 'WHITAKER', 'DECKER', 'YODER', 'COLLIER', 'ZUNIGA', 'CAREY', 'WILCOX',
            'MELENDEZ', 'POOLE', 'ROBERSON', 'LARSEN', 'CONLEY', 'DAVENPORT', 'COPELAND',
            'MASSEY', 'LAM', 'HUFF', 'ROCHA', 'CAMERON', 'JEFFERSON', 'HOOD', 'MONROE',
            'ANTHONY', 'PITTMAN', 'HUYNH', 'RANDALL', 'SINGLETON', 'KIRK', 'COMBS', 'MATHIS',
            'CHRISTIAN', 'SKINNER', 'BRADFORD', 'RICHARD', 'GALVAN', 'WALL', 'BOONE',
            'KIRBY', 'WILKINSON', 'BRIDGES', 'BRUCE', 'ATKINSON', 'CANTRELL', 'REYES',
            'GUNTER', 'TRAN', 'TYLER', 'HANCOCK', 'AGUIRRE', 'MUELLER', 'HERMAN'
        ])
    
    def preprocess_name(self, name: str) -> str:
        """Clean and standardize name before classification."""
        if not name:
            return ""
        
        # Convert to uppercase
        name = str(name).upper().strip()
        
        # Fix common truncations
        truncation_fixes = {
            r'HUMAN SERVIC$': 'HUMAN SERVICES',
            r'COMM DEVELOP$': 'COMMUNITY DEVELOPMENT',
            r'ASSOC$': 'ASSOCIATION',
            r'CORPORAT$': 'CORPORATION',
            r'APARTMEN$': 'APARTMENTS',
            r'PROPERT$': 'PROPERTY',
            r'INVESTMEN$': 'INVESTMENT',
            r'MANAGEMEN$': 'MANAGEMENT',
            r'RESIDENTIA$': 'RESIDENTIAL',
            r'DEVELOP$': 'DEVELOPMENT',
            r'COMMUNIT$': 'COMMUNITY',
            r'AUTHOR$': 'AUTHORITY',
            r'DIST$': 'DISTRICT',
            r'EDUC$': 'EDUCATION',
            r'SERV$': 'SERVICES',
            r'DEPT$': 'DEPARTMENT',
            r'BLDG$': 'BUILDING',
            r'ASSN$': 'ASSOCIATION',
            r'CTR$': 'CENTER',
            r'CRED$': 'CREDIT',
            r'BAPT$': 'BAPTIST',
            r'PENTECOSTES$': 'PENTECOSTES',
            r'ADVENTIS$': 'ADVENTIST',
            r'METHODIS$': 'METHODIST',
            r'PRESBYTERIA$': 'PRESBYTERIAN',
            r'HEARTL$': 'HEARTLAND',
            r'SERVIC$': 'SERVICES',
            r'PUERTA$': 'PUERTA',
            r'COMM$': 'COMMUNITY',
            r'NATIONA$': 'NATIONAL',
            r'INTERNATIONA$': 'INTERNATIONAL',
            r'CHRISTIA$': 'CHRISTIAN',
            r'FOUNDATIO$': 'FOUNDATION',
            r'AMERICA$': 'AMERICAN',
            r'AMER$': 'AMERICA',
            r'DEPART$': 'DEPARTMENT',
            r'SCHO$': 'SCHOOL',
            r'EDUCATIO$': 'EDUCATION',
            r'REHABILITATIO$': 'REHABILITATION',
            r'ASSOCIATIO$': 'ASSOCIATION',
            r'ORGANIZATIO$': 'ORGANIZATION',
            r'ADMINISTRATO$': 'ADMINISTRATOR',
            r'COMMUNICATIO$': 'COMMUNICATION',
            r'TRANSPORTATIO$': 'TRANSPORTATION',
            r'INFORMATIO$': 'INFORMATION',
            r'CONSTRUCTIO$': 'CONSTRUCTION',
            r'DISTRIBUTIO$': 'DISTRIBUTION',
            r'PRODUCTIO$': 'PRODUCTION',
            r'OPERATIO$': 'OPERATION',
            r'RECREATIO$': 'RECREATION',
            r'EDUCATIO$': 'EDUCATION',
            r'FOUNDATIO$': 'FOUNDATION',
            r'CORPORATIO$': 'CORPORATION',
            r'ASSOCIATIO$': 'ASSOCIATION',
            r'FEDERATIO$': 'FEDERATION',
            r'ORGANIZATIO$': 'ORGANIZATION',
            r'INSTITUTIO$': 'INSTITUTION',
            r'COMMISSIO$': 'COMMISSION',
            r'COALITIO$': 'COALITION',
            r'ALLIANC$': 'ALLIANCE',
            r'PARTNERSHI$': 'PARTNERSHIP',
            r'COOPERATIV$': 'COOPERATIVE',
            r'COLLABORATIV$': 'COLLABORATIVE',
            r'INITIATIV$': 'INITIATIVE',
            r'ENTERPRIS$': 'ENTERPRISE',
            r'VENTUR$': 'VENTURE',
            r'PROJEC$': 'PROJECT',
            r'PROGRA$': 'PROGRAM',
            r'SERVIC$': 'SERVICE',
            r'SYSTE$': 'SYSTEM',
            r'NETWOR$': 'NETWORK',
            r'CENTE$': 'CENTER',
            r'OFFIC$': 'OFFICE',
            r'BUREA$': 'BUREAU',
            r'AGENC$': 'AGENCY',
            r'DIVISIO$': 'DIVISION',
            r'BRANC$': 'BRANCH',
            r'SECTIO$': 'SECTION',
            r'DEPARTMEN$': 'DEPARTMENT',
            r'ADMINISTRATI$': 'ADMINISTRATION',
            r'MANAGEMEN$': 'MANAGEMENT',
            r'OPERATIO$': 'OPERATION',
            r'MAINTENANC$': 'MAINTENANCE',
            r'PRESERVATIO$': 'PRESERVATION',
            r'PROTECTIO$': 'PROTECTION',
            r'PREVENTIO$': 'PREVENTION',
            r'INTERVENTIO$': 'INTERVENTION',
            r'REHABILITAT$': 'REHABILITATION',
            r'RESTORATI$': 'RESTORATION',
            r'RENOVATI$': 'RENOVATION',
            r'RECONSTRUCTI$': 'RECONSTRUCTION',
            r'REDEVELOPMEN$': 'REDEVELOPMENT',
            r'REVITALIZATI$': 'REVITALIZATION',
        }
        
        for pattern, replacement in truncation_fixes.items():
            name = re.sub(pattern, replacement, name)
        
        # Fix ETA -> ET AL
        name = re.sub(r'\bETA\b', 'ET AL', name)
        
        # Fix typos
        name = name.replace('COPERATION', 'CORPORATION')
        name = name.replace('CHALLEG', 'CHALLENGE')
        name = name.replace('PENTECOSTES', 'PENTECOSTAL')
        name = name.replace('NEBRSAKA', 'NEBRASKA')
        
        # Standardize spacing around punctuation
        name = re.sub(r'\s*&\s*', ' & ', name)
        name = re.sub(r'\s+-\s+', '-', name)
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip()
    
    def has_hyphenated_name_pattern(self, name: str) -> bool:
        """Check if name has hyphenated surname pattern (common in Hispanic names)."""
        # Pattern for Hispanic double surnames or hyphenated names
        if re.search(r'[A-Z][a-z]+-[A-Z][a-z]+', name):
            return True
        # Check if hyphenated part appears before common first names
        parts = name.split()
        if len(parts) >= 2 and '-' in parts[0]:
            # Check if later parts are common first names
            for part in parts[1:]:
                if part.upper() in self.common_first_names:
                    return True
        return False
    
    def extract_entity_type_ignoring_numbers(self, name: str) -> str:
        """Extract core entity type, ignoring numbers."""
        # Remove numbers but preserve entity markers
        cleaned = re.sub(r'\d+', '', name)
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def calculate_confidence(self, name: str, entity_type: str, indicators: List[str]) -> float:
        """Calculate confidence score based on multiple factors."""
        base_score = 0.1  # Start with low base
        
        # Strong indicators for each type
        strong_patterns = {
            'LLC': [r'\bLLC\b', r'\bL\.L\.C\.', r'\bLIMITED LIABILITY'],
            'Corporation': [r'\bINC\b', r'\bCORP\b', r'\bINCORPORATED\b', r'\bCORPORATION\b'],
            'Trust': [r'\bTRUST\b', r'\bTRUSTEE\b', r'\bLIVING TRUST\b', r'\bREVOCABLE\b'],
            'Government': [r'\bCITY OF\b', r'\bCOUNTY OF\b', r'\bSTATE OF\b', r'\bAUTHORITY\b'],
            'Religious_Nonprofit': [r'\bCHURCH\b', r'\bTEMPLE\b', r'\bMOSQUE\b', r'\bPARISH\b'],
            'Estate': [r'\bESTATE OF\b', r'\bESTATE\b', r'\bDECEASED\b'],
            'Healthcare': [r'\bHOSPITAL\b', r'\bMEDICAL\b', r'\bCLINIC\b'],
            'Bank': [r'\bBANK\b', r'\bCREDIT UNION\b'],
            'Condo': [r'\bCONDOMINIUM\b', r'\bCONDO\b', r'\bHOA\b', r'\bHOMEOWNERS\b'],
        }
        
        # Check for strong indicators
        for pattern in strong_patterns.get(entity_type, []):
            if re.search(pattern, name):
                base_score += 0.4
                break  # Only count once
        
        # Check for strong trust indicators specifically
        if entity_type == 'Trust':
            for pattern in self.strong_trust_indicators:
                if re.search(pattern, name):
                    base_score += 0.3
                    break
        
        # Bonus for multiple corroborating indicators
        if len(indicators) > 2:
            base_score += 0.1 * min(3, len(indicators) - 2)  # Cap bonus at 0.3
        
        # Penalty for very short names (likely truncated)
        if len(name) < 10:
            base_score -= 0.1
        
        # Penalty for ending in truncation pattern
        for pattern in self.truncated_org_patterns:
            if re.search(pattern, name):
                base_score -= 0.1
                break
        
        # Individual-specific scoring
        if entity_type == 'Individual':
            # Check for personal name patterns
            parts = name.split()
            first_name_match = any(part in self.common_first_names for part in parts)
            last_name_match = any(part in self.common_last_names for part in parts)
            
            if first_name_match and last_name_match:
                base_score += 0.5
            elif first_name_match or last_name_match:
                base_score += 0.2
            
            # Check for personal indicators (Jr, Sr, etc.)
            for pattern in self.personal_patterns[:8]:  # First 8 are strongest
                if re.search(pattern, name):
                    base_score += 0.2
                    break
            
            # Penalty if has business indicators
            if any(re.search(p, name) for p in self.business_patterns[:10]):
                base_score -= 0.3
        
        # Corporate/business entity bonus
        elif entity_type in ['Corporation', 'LLC', 'Corporate_Other']:
            # Multiple business patterns increase confidence
            business_count = sum(1 for p in self.business_patterns if re.search(p, name))
            if business_count > 0:
                base_score += min(0.3, business_count * 0.1)
        
        return min(1.0, max(0.0, base_score))
    
    def classify_owner(self, name: str) -> Tuple[str, float, List[str]]:
        """
        Classify owner name with confidence score.
        Returns: (entity_type, confidence, indicators_found)
        """
        if pd.isna(name) or name == "":
            return "Unknown", 0.0, []
        
        # Preprocess name
        name = self.preprocess_name(name)
        upper_name = name.upper()
        
        # For number extraction, clean the name
        name_without_numbers = self.extract_entity_type_ignoring_numbers(upper_name)
        
        indicators = []
        
        # Check for hyphenated Hispanic names
        if self.has_hyphenated_name_pattern(name):
            indicators.append("hyphenated_hispanic_name")
        
        # Check Government patterns with priority
        for pattern in self.gov_patterns:
            if re.search(pattern, upper_name):
                indicators.append("gov_pattern")
                # Strong government indicators
                if any(re.search(p, upper_name) for p in [r'\bCITY OF\b', r'\bCOUNTY OF\b', 
                                                            r'\bSTATE OF\b', r'\bAUTHORITY\b',
                                                            r'\bSCHOOL DISTRICT\b']):
                    confidence = self.calculate_confidence(name, 'Government', indicators)
                    return "Government", max(0.7, confidence), indicators
                break
        
        # Check Religious/Non-profit patterns
        for pattern in self.religious_nonprofit_patterns:
            if re.search(pattern, upper_name):
                indicators.append("religious_nonprofit")
                confidence = self.calculate_confidence(name, 'Religious_Nonprofit', indicators)
                return "Religious_Nonprofit", confidence, indicators
        
        # Check Healthcare patterns
        for pattern in self.healthcare_patterns:
            if re.search(pattern, upper_name):
                indicators.append("healthcare")
                confidence = self.calculate_confidence(name, 'Healthcare', indicators)
                return "Healthcare", confidence, indicators
        
        # Check Bank patterns
        for pattern in self.bank_patterns:
            if re.search(pattern, upper_name):
                indicators.append("bank")
                confidence = self.calculate_confidence(name, 'Bank', indicators)
                return "Bank", confidence, indicators
        
        # Check Condo/HOA patterns
        for pattern in self.condo_patterns:
            if re.search(pattern, upper_name):
                indicators.append("condo")
                confidence = self.calculate_confidence(name, 'Condo', indicators)
                return "Condo", confidence, indicators
        
        # Check Housing patterns
        for pattern in self.housing_patterns:
            if re.search(pattern, upper_name):
                indicators.append("housing")
                # Could be government or non-profit
                if "AUTHORITY" in upper_name or "HOUSING AUTH" in upper_name:
                    return "Government", 0.7, indicators
                else:
                    return "Corporate_Other", 0.5, indicators
        
        # Check LLC patterns (use cleaned name for better matching)
        for pattern in self.llc_patterns:
            if re.search(pattern, name_without_numbers):
                indicators.append("llc_pattern")
                confidence = self.calculate_confidence(name, 'LLC', indicators)
                return "LLC", confidence, indicators
        
        # Check Corporation patterns
        for pattern in self.corp_suffixes:
            if re.search(pattern, name_without_numbers):
                indicators.append("corp_suffix")
                # Additional check for "has_business_suffix"
                if re.search(r'\b(?:INC|CORP|CORPORATION|COMPANY|CO|LTD|LIMITED)\.?(?:\s|$)', upper_name):
                    indicators.append("has_business_suffix")
                confidence = self.calculate_confidence(name, 'Corporation', indicators)
                return "Corporation", confidence, indicators
        
        # Check Trust patterns
        for pattern in self.trust_patterns:
            if re.search(pattern, upper_name):
                indicators.append("trust_pattern")
                confidence = self.calculate_confidence(name, 'Trust', indicators)
                return "Trust", confidence, indicators
        
        # Check Estate patterns
        for pattern in self.estate_patterns:
            if re.search(pattern, upper_name):
                indicators.append("estate_pattern")
                # Check for "REAL ESTATE" which should be corporate
                if re.search(r'\bREAL ESTATE\b', upper_name):
                    for pattern in self.business_patterns:
                        if re.search(pattern, upper_name):
                            indicators.append("business_terms_2")
                            break
                    return "Corporate_Other", 0.4, indicators
                confidence = self.calculate_confidence(name, 'Estate', indicators)
                return "Estate", confidence, indicators
        
        # Check for business patterns (multiple levels)
        business_score = 0
        business_patterns_found = []
        
        # Check primary business patterns
        primary_business = [r'\bPROPERTIES\b', r'\bHOLDINGS\b', r'\bGROUP\b', r'\bPARTNERS\b',
                           r'\bVENTURES\b', r'\bENTERPRISES\b', r'\bDEVELOPMENT\b', 
                           r'\bMANAGEMENT\b', r'\bINVESTMENTS?\b', r'\bCAPITAL\b',
                           r'\bREALTY\b', r'\bREAL ESTATE\b']
        
        for pattern in primary_business:
            if re.search(pattern, upper_name):
                business_score += 1
                business_patterns_found.append("business_terms_1")
                break
        
        # Check secondary business patterns
        for pattern in self.business_patterns:
            if re.search(pattern, upper_name):
                business_score += 0.5
                if "business_terms_2" not in business_patterns_found:
                    business_patterns_found.append("business_terms_2")
                break
        
        # International name patterns
        for pattern_type, pattern in self.international_patterns.items():
            if re.search(pattern, upper_name):
                indicators.append(f"international_{pattern_type}")
        
        # Check if likely individual based on patterns
        parts = upper_name.split()
        
        # Count personal indicators
        personal_score = 0
        
        # Check for common first/last name combinations
        if len(parts) >= 2:
            # Check standard "FIRST LAST" pattern
            if parts[0] in self.common_first_names and parts[-1] in self.common_last_names:
                personal_score += 3
                indicators.append("common_firstname_first")
                indicators.append("common_lastname_last")
            # Check "LAST, FIRST" pattern
            elif parts[0].rstrip(',') in self.common_last_names and len(parts) > 1 and parts[1] in self.common_first_names:
                personal_score += 3
                indicators.append("common_lastname_first")
                indicators.append("common_firstname_second")
            # Check if any part is a common first name
            elif any(part in self.common_first_names for part in parts):
                personal_score += 1
                indicators.append("has_common_firstname")
            # Check if any part is a common last name
            elif any(part in self.common_last_names for part in parts):
                personal_score += 1
                indicators.append("has_common_lastname")
        
        # Check for middle initials (strong individual indicator)
        if re.search(r'\b[A-Z]\.\s*(?:[A-Z]|$)', upper_name) or re.search(r'\b[A-Z]\s+[A-Z]\s+', upper_name):
            personal_score += 1
            indicators.append("has_middle_initial")
        
        # Check for personal patterns (Jr, Sr, etc.)
        for pattern in self.personal_patterns:
            if re.search(pattern, upper_name):
                personal_score += 1.5
                indicators.append("personal_pattern_1")
                break
        
        # Check for conjunctions suggesting multiple people
        if re.search(r'\b(?:AND|&|OR)\b', upper_name):
            indicators.append("has_conjunction")
            # This could be personal or business
            if personal_score > 0:
                personal_score += 0.5
        
        # Check for numbers (often in business names)
        if re.search(r'\d', upper_name):
            indicators.append("contains_numbers")
            business_score += 0.25
        
        # Make classification decision
        if business_score >= 1 and personal_score < 2:
            indicators.extend(business_patterns_found)
            confidence = self.calculate_confidence(name, 'Corporate_Other', indicators)
            return "Corporate_Other", confidence, indicators
        elif personal_score >= 2:
            confidence = self.calculate_confidence(name, 'Individual', indicators)
            return "Individual", confidence, indicators
        elif business_score > 0:
            indicators.extend(business_patterns_found)
            confidence = self.calculate_confidence(name, 'Corporate_Other', indicators)
            return "Corporate_Other", confidence, indicators
        elif personal_score > 0:
            confidence = self.calculate_confidence(name, 'Individual', indicators)
            return "Individual", confidence, indicators
        
        # Check for truncated patterns as last resort
        for pattern in self.truncated_org_patterns:
            if re.search(pattern, upper_name):
                indicators.append("truncated_org")
                return "Corporate_Other", 0.3, indicators
        
        # Default to Unknown
        return "Unknown", 0.0, indicators

# ── Main Analysis Function ─────────────────────────────────────────────
def analyze_owner_classifications(gdb_path: str, parcels_fc: str, output_dir: str):
    """Main analysis function."""
    print("Enhanced Absentee Residential Owner Classification Analysis")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parcels
    print("\nLoading parcels from geodatabase...")
    fields = ["*"]  # Get all fields
    
    try:
        # Read into pandas dataframe
        data = []
        field_names = None
        
        with arcpy.da.SearchCursor(parcels_fc, fields) as cursor:
            field_names = cursor.fields
            for row in cursor:
                data.append(row)
        
        df = pd.DataFrame(data, columns=field_names)
        print(f"  Loaded {len(df):,} parcels")
        
        # Show available fields for debugging
        print("\n  Available fields in dataset:")
        addr_related_fields = [f for f in field_names if any(term in f.lower() for term in 
                              ['addr', 'add', 'street', 'mail', 'zip', 'owner', 'ow', 'situs'])]
        for field in sorted(addr_related_fields):
            non_null = df[field].notna().sum()
            print(f"    {field}: {non_null:,} non-null values")
        
    except Exception as e:
        print(f"Error loading parcels: {e}")
        return
    
    # Identify absentee residential properties
    df = identify_absentee_residential(
        df, 
        PARCEL_ADDR_FIELDS,
        OWNER_ADDR_FIELDS,
        LAND_USE_FIELD,
        RESIDENTIAL_CODES
    )
    
    # Filter to absentee residential only
    absentee_df = df[df['IsAbsenteeResidential']].copy()
    print(f"\nAnalyzing {len(absentee_df):,} absentee residential properties")
    
    if len(absentee_df) == 0:
        print("No absentee residential properties found!")
        return
    
    # Find owner name field
    owner_field = None
    for field in OWNER_NAME_FIELDS:
        if field in absentee_df.columns:
            owner_field = field
            print(f"Using owner name field: {owner_field}")
            break
    
    if not owner_field:
        print("Error: No owner name field found!")
        return
    
    # Initialize classifier
    classifier = EnhancedOwnerNameClassifier()
    
    # Classify owners
    print("\nClassifying owner names...")
    classifications = []
    confidence_scores = []
    indicators_list = []
    
    for idx, row in absentee_df.iterrows():
        owner_name = row[owner_field]
        entity_type, confidence, indicators = classifier.classify_owner(owner_name)
        classifications.append(entity_type)
        confidence_scores.append(confidence)
        indicators_list.append(indicators)
        
        if idx % 5000 == 0:
            print(f"  Processed {idx:,} records...")
    
    absentee_df['ClassifiedType'] = classifications
    absentee_df['ClassificationConfidence'] = confidence_scores
    absentee_df['ClassificationIndicators'] = indicators_list
    
    # Analysis results
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS - ABSENTEE RESIDENTIAL PROPERTIES")
    print("=" * 60)
    
    # Overall statistics
    type_counts = absentee_df['ClassifiedType'].value_counts()
    print("\nEntity Type Distribution:")
    print("-" * 40)
    total = len(absentee_df)
    for entity_type, count in type_counts.items():
        pct = count / total * 100
        print(f"{entity_type:20} {count:8,} ({pct:5.1f}%)")
    
    # Confidence analysis
    print("\nConfidence Score Statistics:")
    print("-" * 40)
    for entity_type in type_counts.index:
        mask = absentee_df['ClassifiedType'] == entity_type
        if mask.any():
            scores = absentee_df.loc[mask, 'ClassificationConfidence']
            print(f"{entity_type:20} "
                  f"Mean: {scores.mean():.3f}, "
                  f"Median: {scores.median():.3f}, "
                  f"Min: {scores.min():.3f}, "
                  f"Max: {scores.max():.3f}")
    
    # Low confidence analysis
    low_conf_threshold = 0.5
    low_conf_mask = absentee_df['ClassificationConfidence'] < low_conf_threshold
    low_conf_count = low_conf_mask.sum()
    print(f"\nLow Confidence Classifications (< {low_conf_threshold}): "
          f"{low_conf_count:,} ({low_conf_count/total*100:.1f}%)")
    
    # Distance analysis if available
    if DISTANCE_FIELD in absentee_df.columns:
        print("\n" + "=" * 60)
        print("DISTANCE ANALYSIS BY ENTITY TYPE")
        print("=" * 60)
        
        # Convert distance to numeric, handling any errors
        absentee_df[DISTANCE_FIELD] = pd.to_numeric(absentee_df[DISTANCE_FIELD], errors='coerce')
        
        # Remove invalid distances
        valid_dist_mask = absentee_df[DISTANCE_FIELD].notna() & (absentee_df[DISTANCE_FIELD] >= 0)
        dist_df = absentee_df[valid_dist_mask].copy()
        
        if len(dist_df) > 0:
            print(f"\nAnalyzing {len(dist_df):,} properties with valid distances")
            print("-" * 60)
            
            for entity_type in type_counts.index:
                mask = dist_df['ClassifiedType'] == entity_type
                if mask.any():
                    distances = dist_df.loc[mask, DISTANCE_FIELD]
                    if len(distances) > 0:
                        print(f"\n{entity_type}:")
                        print(f"  Count: {len(distances):,}")
                        print(f"  Mean Distance: {distances.mean():.1f} km")
                        print(f"  Median Distance: {distances.median():.1f} km")
                        print(f"  25th Percentile: {distances.quantile(0.25):.1f} km")
                        print(f"  75th Percentile: {distances.quantile(0.75):.1f} km")
                        print(f"  Max Distance: {distances.max():.1f} km")
                        
                        # Local vs distant
                        local = (distances < 50).sum()
                        distant = (distances >= 50).sum()
                        if len(distances) > 0:
                            print(f"  Local (<50km): {local:,} ({local/len(distances)*100:.1f}%)")
                            print(f"  Distant (≥50km): {distant:,} ({distant/len(distances)*100:.1f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Entity type distribution
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.title('Absentee Residential Property Owner Types')
    plt.xlabel('Entity Type')
    plt.ylabel('Number of Properties')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_type_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Confidence score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(absentee_df['ClassificationConfidence'], bins=50, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Low Confidence Threshold')
    plt.title('Classification Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Properties')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Distance boxplot by entity type (if available)
    if DISTANCE_FIELD in absentee_df.columns and len(dist_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        plot_data = []
        plot_labels = []
        
        for entity_type in type_counts.index:
            mask = dist_df['ClassifiedType'] == entity_type
            if mask.any():
                distances = dist_df.loc[mask, DISTANCE_FIELD].dropna()
                if len(distances) > 0:
                    plot_data.append(distances)
                    plot_labels.append(f"{entity_type}\n(n={len(distances):,})")
        
        if plot_data:
            plt.boxplot(plot_data, labels=plot_labels)
            plt.title('Owner Distance Distribution by Entity Type')
            plt.ylabel('Distance (km)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'distance_by_entity_type.png'), dpi=300)
            plt.close()
    
    # Save detailed results
    print("\nSaving detailed results...")
    
    # 1. Summary statistics
    summary_file = os.path.join(output_dir, 'classification_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Enhanced Absentee Residential Owner Classification Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Parcels Analyzed: {len(df):,}\n")
        f.write(f"Residential Parcels: {df['IsResidential'].sum():,}\n")
        f.write(f"Owner-Occupied: {df['IsOwnerOccupied'].sum():,}\n")
        f.write(f"Absentee Residential: {len(absentee_df):,}\n\n")
        
        f.write("Entity Type Distribution:\n")
        f.write("-" * 40 + "\n")
        for entity_type, count in type_counts.items():
            pct = count / total * 100
            f.write(f"{entity_type:20} {count:8,} ({pct:5.1f}%)\n")
        
        f.write("\nConfidence Score Statistics:\n")
        f.write("-" * 40 + "\n")
        for entity_type in type_counts.index:
            mask = absentee_df['ClassifiedType'] == entity_type
            if mask.any():
                scores = absentee_df.loc[mask, 'ClassificationConfidence']
                f.write(f"{entity_type:20} "
                       f"Mean: {scores.mean():.3f}, "
                       f"Median: {scores.median():.3f}, "
                       f"Min: {scores.min():.3f}, "
                       f"Max: {scores.max():.3f}\n")
    
    # 2. Sample classifications for each type
    samples_file = os.path.join(output_dir, 'classification_samples.csv')
    sample_data = []
    
    for entity_type in type_counts.index:
        mask = absentee_df['ClassifiedType'] == entity_type
        if mask.any():
            # Get up to 20 samples, prioritizing different confidence levels
            type_df = absentee_df[mask].copy()
            type_df = type_df.sort_values('ClassificationConfidence', ascending=False)
            
            # Take samples from different confidence ranges
            high_conf = type_df[type_df['ClassificationConfidence'] >= 0.7].head(7)
            mid_conf = type_df[(type_df['ClassificationConfidence'] >= 0.4) & 
                              (type_df['ClassificationConfidence'] < 0.7)].head(7)
            low_conf = type_df[type_df['ClassificationConfidence'] < 0.4].head(6)
            
            for df_subset in [high_conf, mid_conf, low_conf]:
                for _, row in df_subset.iterrows():
                    sample_data.append({
                        'EntityType': entity_type,
                        'OwnerName': row[owner_field],
                        'Confidence': row['ClassificationConfidence'],
                        'Indicators': ', '.join(row['ClassificationIndicators']),
                        'Distance_km': row.get(DISTANCE_FIELD, 'N/A')
                    })
    
    pd.DataFrame(sample_data).to_csv(samples_file, index=False)
    
    # 3. Low confidence classifications for review
    low_conf_file = os.path.join(output_dir, 'low_confidence_classifications.txt')
    with open(low_conf_file, 'w') as f:
        f.write("LOW CONFIDENCE CLASSIFICATIONS FOR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file contains owner names with classification confidence < 0.5\n")
        f.write("Format: [Confidence] EntityType | OwnerName | Indicators\n")
        f.write("-" * 80 + "\n\n")
        
        low_conf_df = absentee_df[low_conf_mask].copy()
        low_conf_df = low_conf_df.sort_values(['ClassifiedType', 'ClassificationConfidence'])
        
        current_type = None
        for _, row in low_conf_df.iterrows():
            if current_type != row['ClassifiedType']:
                current_type = row['ClassifiedType']
                count = (low_conf_df['ClassifiedType'] == current_type).sum()
                f.write(f"\n{current_type.upper()} ({count} records)\n")
                f.write("-" * 60 + "\n")
            
            f.write(f"[{row['ClassificationConfidence']:.3f}] "
                   f"{row['ClassifiedType']:15} | "
                   f"{str(row[owner_field])[:50]:50} | "
                   f"{', '.join(row['ClassificationIndicators'])}\n")
    
    # 4. Full classification results
    results_file = os.path.join(output_dir, 'full_classification_results.csv')
    export_cols = [owner_field, 'ClassifiedType', 'ClassificationConfidence', 
                   'IsResidential', 'IsOwnerOccupied', 'IsAbsenteeResidential']
    if DISTANCE_FIELD in absentee_df.columns:
        export_cols.append(DISTANCE_FIELD)
    
    absentee_df[export_cols].to_csv(results_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - Summary: {os.path.basename(summary_file)}")
    print(f"  - Samples: {os.path.basename(samples_file)}")
    print(f"  - Low confidence: {os.path.basename(low_conf_file)}")
    print(f"  - Full results: {os.path.basename(results_file)}")

# ── Main Execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set up ArcGIS environment
    arcpy.env.overwriteOutput = True
    
    # Run analysis
    analyze_owner_classifications(GDB, PARCELS_FC, OUTPUT_DIR)
    
    match1 = re.match(pattern, std_addr1)
    match2 = re.match(pattern, std_addr2)
    
    if match1 and match2:
        num1, street1 = match1.groups()
        num2, street2 = match2.groups()
        
        # Numbers must match exactly
        if num1 == num2:
            # Clean up street names
            street1 = street1.strip() if street1 else ""
            street2 = street2.strip() if street2 else ""
            
            # Check if street names are similar enough
            if street1 and street2:
                # Exact match
                if street1 == street2:
                    return True
                
                # Check if one starts with the other (handles abbreviations)
                if street1.startswith(street2) or street2.startswith(street1):
                    return True
                
                # Check first word match (handles different street types)
                words1 = street1.split()
                words2 = street2.split()
                if words1 and words2 and words1[0] == words2[0]:
                    return True
    
    # Alternative pattern for addresses without clear street type
    match1 = re.match(r'^(\d+)\s+(\S+)', std_addr1)
    match2 = re.match(r'^(\d+)\s+(\S+)', std_addr2)
    
    if match1 and match2 and match1.group(1) == match2.group(1):
        # Same house number, check if first word of street matches
        if match1.group(2) == match2.group(2):
            return True
    
    # Check ZIP codes if provided and addresses are somewhat similar
    if pd.notna(zip1) and pd.notna(zip2):
        try:
            zip1_str = str(int(float(str(zip1))))[:5]
            zip2_str = str(int(float(str(zip2))))[:5]
            
            # If ZIPs match and addresses share the same number, likely a match
            if len(zip1_str) >= 5 and len(zip2_str) >= 5 and zip1_str == zip2_str:
                num_match1 = re.match(r'^(\d+)', std_addr1)
                num_match2 = re.match(r'^(\d+)', std_addr2)
                if num_match1 and num_match2 and num_match1.group(1) == num_match2.group(1):
                    return True
        except:
            pass
    
    return False

def identify_absentee_residential(df: pd.DataFrame, 
                                   parcel_addr_fields: List[str],
                                   owner_addr_fields: List[str],
                                   land_use_field: str,
                                   residential_codes: List[int]) -> pd.DataFrame:
    """Identify absentee residential properties."""
    print("\nIdentifying absentee residential properties...")
    
    # First identify residential properties
    if land_use_field in df.columns:
        df[land_use_field] = pd.to_numeric(df[land_use_field], errors='coerce')
        df['IsResidential'] = df[land_use_field].isin(residential_codes)
        residential_count = df['IsResidential'].sum()
        print(f"  Found {residential_count:,} residential properties")
    else:
        print(f"  Warning: Land use field '{land_use_field}' not found - analyzing all properties")
        df['IsResidential'] = True
        residential_count = len(df)
    
    # Find best address fields to use
    parcel_addr_field = None
    for field in parcel_addr_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            if non_null > 0:
                parcel_addr_field = field
                print(f"  Using parcel address field: {parcel_addr_field} ({non_null:,} non-null values)")
                break
    
    owner_addr_field = None
    for field in owner_addr_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            if non_null > 0:
                owner_addr_field = field
                print(f"  Using owner address field: {owner_addr_field} ({non_null:,} non-null values)")
                break
    
    # Find ZIP fields
    parcel_zip_field = PARCEL_ZIP if PARCEL_ZIP in df.columns else None
    owner_zip_field = None
    for field in OWNER_ZIP_FIELDS:
        if field in df.columns:
            owner_zip_field = field
            break
    
    # Check for owner-occupied
    if parcel_addr_field and owner_addr_field:
        print(f"  Checking address matching...")
        
        # Create a debug sample to check matching
        debug_sample = df[df['IsResidential']].head(1000)
        debug_matches = 0
        
        df['AddressMatch'] = False
        
        # Process in chunks for better performance
        chunk_size = 10000
        total_residential = df['IsResidential'].sum()
        
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk_indices = df.index[i:chunk_end]
            chunk_df = df.loc[chunk_indices]
            
            # Get residential properties in this chunk
            res_mask = chunk_df['IsResidential']
            res_indices = chunk_indices[res_mask]
            
            if len(res_indices) > 0:
                # Apply address matching to residential properties
                for idx in res_indices:
                    p_addr = df.loc[idx, parcel_addr_field]
                    o_addr = df.loc[idx, owner_addr_field]
                    p_zip = df.loc[idx, parcel_zip_field] if parcel_zip_field else None
                    o_zip = df.loc[idx, owner_zip_field] if owner_zip_field else None
                    
                    if addresses_match(p_addr, o_addr, p_zip, o_zip):
                        df.loc[idx, 'AddressMatch'] = True
            
            if i % 50000 == 0 and i > 0:
                print(f"    Processed {i:,} records...")
        
        df['IsOwnerOccupied'] = df['IsResidential'] & df['AddressMatch']
        owner_occupied_count = df['IsOwnerOccupied'].sum()
        print(f"  Found {owner_occupied_count:,} owner-occupied residential properties")
        
        # Debug info
        if owner_occupied_count < 50000 and residential_count > 100000:
            print("\n  Warning: Low owner-occupied count. Checking sample addresses:")
            sample = df[df['IsResidential']].head(5)
            for idx, row in sample.iterrows():
                p_addr = row.get(parcel_addr_field, "")
                o_addr = row.get(owner_addr_field, "")
                print(f"    Parcel: {p_addr}")
                print(f"    Owner:  {o_addr}")
                print(f"    Match:  {addresses_match(p_addr, o_addr)}")
                print()
    else:
        print("  Warning: Address fields not found - cannot identify owner-occupied")
        df['IsOwnerOccupied'] = False
        owner_occupied_count = 0
    
    # Identify absentee residential
    df['IsAbsenteeResidential'] = df['IsResidential'] & ~df['IsOwnerOccupied']
    absentee_count = df['IsAbsenteeResidential'].sum()
    
    print(f"\n  Summary:")
    print(f"    Total properties: {len(df):,}")
    print(f"    Residential: {residential_count:,}")
    print(f"    Owner-occupied: {owner_occupied_count:,}")
    print(f"    Absentee residential: {absentee_count:,}")
    
    if residential_count > 0:
        absentee_rate = absentee_count / residential_count * 100
        print(f"    Absentee rate: {absentee_rate:.1f}%")
    
    return df

# ── Enhanced Owner Name Classifier ─────────────────────────────────────
class EnhancedOwnerNameClassifier:
    """Enhanced classifier with improved pattern matching for residential property owners."""
    
    def __init__(self):
        # Expanded corporate indicators
        self.corp_suffixes = [
            r'\bINC\.?(?:\s|$)', r'\bINCORPORATED\b', r'\bCORP\.?(?:\s|$)', r'\bCORPORATION\b',
            r'\bCOMPANY\b', r'\bCO\.(?:\s|$)', r'\bLTD\.?(?:\s|$)', r'\bLIMITED\b',
            r'\bPLC\b', r'\bSA\b', r'\bAG\b', r'\bGMBH\b', r'\bNV\b', r'\bBV\b',
            r'\bCORPS?\b', r'\bINCORP\b', r'\bCORPORATED\b', r'\bCOMPANIES\b',
            r'\bCORPORAT(?:$|[^E])'  # For CORPORAT (truncated)
        ]
        
        # Enhanced LLC patterns
        self.llc_patterns = [
            r'\bLLC\.?(?:\s|$)', r'\bL\.L\.C\.(?:\s|$)', r'\bLLP\.?(?:\s|$)', 
            r'\bL\.L\.P\.(?:\s|$)', r'\bLP\.?(?:\s|$)', r'\bL\.P\.(?:\s|$)',
            r'\bLIMITED LIABILITY', r'\bLIMITED PARTNERSHIP', r'\bLTD LIABILITY\b',
            r'\bLIMITED LIAB\b', r'\bL L C\b', r'\bL C\b', r'\bLMT PART\b',
            r'\bLTD PARTNERSHIP\b', r'\bLIMITED PARTNERS\b'
        ]
        
        # Strong trust indicators
        self.strong_trust_indicators = [
            r'\bTR\s+(?:DTD|DATED)',  # TR DATED
            r'\bREVOCABLE.*TRUST',
            r'\bLIVING.*TRUST',
            r'\bFAMILY.*TRUST',
            r'\b(?:TRUSTEE|TRUSTEES)\s+(?:OF|FOR)\b',
        ]
        
        # Expanded trust patterns
        self.trust_patterns = [
            r'\bTRUST\b', r'\bTRUSTEE[S]?\b', r'\bREVOCABLE\b', r'\bIRREVOCABLE\b',
            r'\bLIVING TRUST\b', r'\bFAMILY TRUST\b', r'\b(?:TR|TRS|TRST)\b',
            r'\bTRST\b', r'\bREV TR\b', r'\bFAM TR\b', r'\bQTIP\b', r'\bCHARITABLE\b',
            r'\bTESTAMENTARY\b', r'\bGRANTOR\b', r'\bSURVIVOR[S]? TRUST\b',
            r'\bREV TRUST\b', r'\bREVOC TR\b', r'\bIRREV TR\b', r'\bU/A\b', r'\bU/T\b',
            r'\bUNDER AGREEMENT\b', r'\bUNDER TRUST\b', r'\b(?:DTD|DATED)\b',
            r'\bTRU\b', r'\bLIV TR\b', r'\bLIVING TR\b'
        ]
        
        # Estate patterns
        self.estate_patterns = [
            r'\bESTATE\b', r'\bEST\.(?:\s|$)', r'\bESTATE OF\b', r'\bDECEASED\b',
            r'\bEXECUTOR\b', r'\bADMINISTRATOR\b', r'\bHEIRS\b', r'\bSURVIVOR\b',
            r'\bESTATES\b', r'\bEST OF\b', r'\bHEIRS OF\b', r'\bSURVIVING\b',
            r'\bPERSONAL REP\b', r'\bPERS REP\b', r'\bEXEC\b', r'\bADMIN\b'
        ]
        
        # Government/Public patterns
        self.gov_patterns = [
            r'\bCITY OF\b', r'\bCOUNTY OF\b', r'\bSTATE OF\b', r'\bUNITED STATES\b',
            r'\bDEPT\.?(?:\s|$)', r'\bDEPARTMENT\b', r'\bAUTHORITY\b', r'\bMUNICIPAL\b',
            r'\bSCHOOL DISTRICT\b', r'\bHOUSING AUTHORITY\b', r'\bPUBLIC\b',
            r'\bGOVERNMENT\b', r'\bFEDERAL\b', r'\bUSA\b', r'\bU\.S\.A\.(?:\s|$)',
            r'\bUNIVERSITY\b', r'\bCOLLEGE\b', r'\bBOARD OF\b', r'\bCOMMISSION\b',
            r'\bAGENCY\b', r'\bDISTRICT\b', r'\bTOWNSHIP\b', r'\bVILLAGE OF\b',
            r'\bHOUSING AUTH\b', r'\bSCH DIST\b', r'\bSCHOOL DIST\b', r'\bSCHL DIST\b',
            r'\bCNTY\b', r'\bCOUNTY\b', r'\bFIRE DIST\b', r'\bFIRE RESCUE\b',
            r'\bSECRETARY\b.*\bHOUSING\b', r'\bHUD\b', r'\bREGENTS\b', r'\bBOARD\b.*\bREGENTS\b',
            r'\bPONCA TRIBE\b', r'\bTRIBE OF\b'
        ]
        
        # Religious/Non-profit patterns - expanded for residential area churches
        self.religious_nonprofit_patterns = [
            r'\bCHURCH\b', r'\bCHURCHES\b', r'\bCATHOLIC\b', r'\bBAPTIST\b', 
            r'\bLUTHERAN\b', r'\bMETHODIST\b', r'\bPRESBYTERIAN\b', r'\bEPISCOPAL\b',
            r'\bCHRISTIAN\b', r'\bCHRIST\b', r'\bJESUS\b', r'\bGOSPEL\b', 
            r'\bMINISTRY\b', r'\bMINISTRIES\b', r'\bTEMPLE\b', r'\bSYNAGOGUE\b', 
            r'\bMOSQUE\b', r'\bPARISH\b', r'\bCONGREGATION\b', r'\bJEHOVAH\b', 
            r'\bSAINT[S]?\b', r'\bST\.?\s', r'\bCHAPEL\b', r'\bFAITH\b',
            r'\bIGLESIA\b', r'\bCENTRO CRISTIANO\b', r'\bMINISTERIO\b',  # Spanish
            r'\bCHABAD\b', r'\bLUBAVITCH\b',  # Jewish
            r'\bPENTECOSTE[S]?\b', r'\bASSEMBLY OF GOD\b',
            r'\bHABITAT\s+FOR\s+HUMANITY\b', r'\bYMCA\b', r'\bYWCA\b', r'\bFOUNDATION\b',
            r'\bINSTITUTE\b', r'\bCHARITY\b', r'\bCHARITABLE\b', r'\bNONPROFIT\b', 
            r'\bNON-PROFIT\b', r'\bSOCIETY\b', r'\bASSOCIATION\b', r'\bLEAGUE\b',
            r'\bCOMMUNITY\s+CENTER\b', r'\bSALVATION ARMY\b', r'\bMISSION\b',
            r'\bRESCUE MISSION\b', r'\bHOPE MISSION\b'
        ]
        
        # Healthcare patterns
        self.healthcare_patterns = [
            r'\bHOSPITAL\b', r'\bMEDICAL\b', r'\bHEALTH\b', r'\bCLINIC\b',
            r'\bHEALTHCARE\b', r'\bHEALTH CENTER\b', r'\bNURSING\b', 
            r'\bHOSPICE\b', r'\bREHAB\b', r'\bREHABILITATION\b',
            r'\bPHYSICIANS?\b', r'\bDIALYSIS\b', r'\bANIMAL HEALTH\b'
        ]
        
        # Banking/Financial patterns
        self.bank_patterns = [
            r'\bBANK\b', r'\bCREDIT UNION\b', r'\bSAVINGS\b', r'\bFINANCIAL\b',
            r'\bBANKING\b', r'\bFIRST NATIONAL\b', r'\bNATIONAL BANK\b',
            r'\bBANK OF\b', r'\bBANK & TRUST\b', r'\bBANK AND TRUST\b'
        ]
        
        # Condominium/HOA patterns
        self.condo_patterns = [
            r'\bCONDO\b', r'\bCONDOMINIUM\b', r'\bCONDOMINION\b',
            r'\bCONDO ASSOC\b', r'\bCONDOMINIUM ASSOC\b', r'\bCONDOMINIUM ASSN\b',
            r'\bHOMEOWNERS?\b', r'\bHOA\b', r'\bTOWNHOME\b', r'\bTOWNHOUSE\b',
            r'\bOWNERS ASSOC\b', r'\bOWNERS ASSN\b'
        ]
        
        # Housing-specific patterns
        self.housing_patterns = [
            r'HOUSING AUTHOR',  # Housing Authority (truncated)
            r'SENIOR COTTAGES?',
            r'SENIOR HOUSING',
            r'AFFORDABLE HOUSING',
            r'GOOD SAMARITAN',
            r'COTTAGES?\b',
            r'SENIOR LIVING',
            r'ASSISTED LIVING',
            r'RETIREMENT'
        ]
        
        # Truncated organization patterns
        self.truncated_org_patterns = [
            r'HUMAN SERVIC$',  # Human Services
            r'NATURAL RESOUR$',  # Natural Resources
            r'COMM DEVELOP$',  # Community Development
            r'ASSOC?I?A?$',  # Association variants
            r'DEVELOP$',  # Development
            r'CORPORAT$',  # Corporation
            r'APARTMEN$',  # Apartments
            r'PROPERT$',  # Properties/Property
            r'COMMUNIT$',  # Community
            r'MANAGEMEN$',  # Management
            r'INVESTMEN$',  # Investment
            r'RESIDENTIA$',  # Residential
        ]
        
        # Business patterns - focused on residential property owners
        self.business_patterns = [
            r'\bPROPERTIES\b', r'\bHOLDINGS\b', r'\bGROUP\b', r'\bPARTNERS\b',
            r'\bVENTURES\b', r'\bENTERPRISES\b', r'\bDEVELOPMENT\b', r'\bMANAGEMENT\b',
            r'\bINVESTMENTS?\b', r'\bCAPITAL\b', r'\bREALTY\b', r'\bREAL ESTATE\b',
            r'\bASSET[S]?\b', r'\bEQUITY\b', r'\bFUND[S]?\b', r'\bPORTFOLIO\b',
            r'\bACQUISITIONS?\b', r'\bCONSTRUCTION\b', r'\bBUILDERS?\b',
            r'\bASSOCIATES\b', r'\bSERVICES\b', r'\bSOLUTIONS\b', r'\bCENTER\b',
            r'\bRENTALS?\b', r'\bLEASING\b', r'\bHOMES?\b', r'\bLAND\b', r'\bPROPERTY\b',
            r'\bESTATES\b', r'\bRESIDENTIAL\b', r'\bAPARTMENTS?\b', r'\bBROTHERS\b',
            r'\bFAMILY\b', r'\bPARTNERSHIP\b', r'\bJOINT VENTURE\b',
            r'\bMANOR\b', r'\bVILLAGE\b', r'\bCOMMONS\b', r'\bCROSSING\b',
            r'\bPARK\b', r'\bGARDENS?\b', r'\bPLACE\b', r'\bSQUARE\b',
            r'\bPOINTE?\b', r'\bRIDGE\b', r'\bCREEK\b', r'\bLAKE\b', r'\bHILLS?\b',
            r'\bVALLEY\b', r'\bVISTA\b', r'\bVIEW\b', r'\bHEIGHTS\b', r'\bMEADOWS?\b'
        ]
        
        # Personal name indicators - enhanced with ETA pattern
        self.personal_patterns = [
            r'\bJR\.?(?:\s|$)', r'\bSR\.?(?:\s|$)', r'\bIII\b', r'\bII\b', r'\bIV\b',
            r'\bMR\.?(?:\s|$)', r'\bMRS\.?(?:\s|$)', r'\bMS\.?(?:\s|$)', r'\bDR\.?(?:\s|$)',
            r'\bAND\b', r'\b&\b', r'\bOR\b', r'\bETAL\.?(?:\s|$)',
            r'\bET AL\.?(?:\s|$)', r'\bET UX\.?(?:\s|$)', r'\bET VIR\.?(?:\s|$)',
            r'\bETA\b',  # Common misspelling/truncation of ET AL
            r'\bHUSBAND\b', r'\bWIFE\b', r'\bSPOUSE\b', r'\bMARRIED\b',
            r'\bJOINT\b', r'\bTENANTS\b', r'\bJT TEN\b', r'\bJTWROS\b'
        ]
        
        # International name patterns
        self.international_patterns = {
            'indian': r'\b(?:REDDY|KUMAR|SINGH|PATEL|SHARMA|GUPTA|SHAH|RAO|NAIR|IYER)\b',
            'asian': r'\b(?:NGUYEN|TRAN|LE|PHAM|HOANG|HO|NGO|VU|DANG|BUI|DO|PHAN|DUONG|LAM)\b',
            'hispanic_special': r'\b(?:DE LA|DE LOS|DE LAS|DEL|DELA)\b',
        }
        
        # Common first names - pruned for likely property owners (older, predominantly male)
        self.common_first_names = set([
            # Traditional male names - more common among property owners
            'JOHN', 'DAVID', 'MICHAEL', 'ROBERT', 'JAMES', 'WILLIAM', 'RICHARD', 'THOMAS',
            'JOSEPH', 'CHARLES', 'CHRISTOPHER', 'DANIEL', 'MATTHEW', 'DONALD', 'KENNETH',
            'MARK', 'PAUL', 'STEVEN', 'ANDREW', 'GEORGE', 'EDWARD', 'RONALD', 'TIMOTHY',
            'GARY', 'LARRY', 'JEFFREY', 'FRANK', 'SCOTT', 'ERIC', 'STEPHEN', 'RAYMOND',
            'GREGORY', 'SAMUEL', 'BENJAMIN', 'PATRICK', 'JACK', 'DENNIS', 'JERRY', 'CARL',
            'HAROLD', 'DOUGLAS', 'HENRY', 'ARTHUR', 'PETER', 'ALBERT', 'WILLIE', 'WAYNE',
            'RALPH', 'EUGENE', 'RUSSELL', 'ROY', 'LOUIS', 'PHILIP', 'JOHNNY', 'ERNEST',
            'MARTIN', 'CRAIG', 'ALAN', 'CLARENCE', 'WALTER', 'CARL', 'BRUCE', 'KEITH',
            'NORMAN', 'MARVIN', 'VINCENT', 'GLENN', 'HOWARD', 'FRED', 'LEON', 'DALE',
            'CLIFFORD', 'CHESTER', 'LLOYD', 'MELVIN', 'VERNON', 'HERMAN', 'LEROY',
            'FRANCIS', 'STANLEY', 'LEONARD', 'NATHAN', 'JOEL', 'EDWIN', 'DON', 'GORDON',
            'DEAN', 'JEROME', 'JIM', 'TOM', 'BILL', 'MIKE', 'DAVE', 'RICK', 'BOB',
            'STEVE', 'JEFF', 'ROGER', 'TERRY', 'BRUCE', 'RANDY', 'BERNARD', 'NICHOLAS',
            'LEE', 'ALLEN', 'KEVIN', 'BRIAN', 'JOSHUA', 'RYAN', 'JACOB', 'JASON',
            'JUSTIN', 'BRANDON', 'JOSE', 'JUAN', 'MIGUEL', 'CARLOS', 'ANTONIO', 'MANUEL',
            'PEDRO', 'FRANCISCO', 'JESUS', 'LUIS', 'JORGE', 'ROBERTO', 'ALBERTO', 'FERNANDO',
            'PABLO', 'MARCO', 'RICARDO', 'EDUARDO', 'JAVIER', 'SERGIO', 'ARTURO', 'ALEJANDRO',
            'ENRIQUE', 'GERARDO', 'MARIO', 'ADRIAN', 'HECTOR', 'RAFAEL', 'OMAR', 'RAUL',
            'ARMANDO', 'ALFREDO', 'MARTIN', 'FELIX', 'ANDRES', 'JULIO', 'DIEGO', 'IVAN',
            'VICTOR', 'SAMUEL', 'GABRIEL', 'ALEXANDER', 'ANTHONY', 'MARCUS', 'MATTHEW',
            'LAWRENCE', 'WARREN', 'HERBERT', 'FREDERICK', 'CLIFTON', 'WALLACE', 'STANLEY',
            'EUGENE', 'RALPH', 'HOWARD', 'KENNETH', 'HAROLD', 'FRANCIS', 'BENJAMIN',
            'HARVEY', 'ARNOLD', 'GERALD', 'KARL', 'NEIL', 'STUART', 'MARSHALL', 'GUY',
            'GILBERT', 'ROLAND', 'THEODORE', 'BERNARD', 'EUGENE', 'HERBERT', 'NORMAN',
            'CLIFFORD', 'RAYMOND', 'WARREN', 'HOWARD', 'STANLEY', 'ARTHUR', 'ERNEST',
            'LEONARD', 'ALBERT', 'LAWRENCE', 'GORDON', 'NORMAN', 'RUSSELL', 'WALLACE',
            
            # Common female names - property owners (reduced set)
            'MARY', 'PATRICIA', 'LINDA', 'BARBARA', 'ELIZABETH', 'JENNIFER', 'MARIA',
            'SUSAN', 'MARGARET', 'DOROTHY', 'LISA', 'NANCY', 'KAREN', 'BETTY', 'HELEN',
            'SANDRA', 'DONNA', 'CAROL', 'RUTH', 'SHARON', 'MICHELLE', 'LAURA', 'SARAH',
            'KIMBERLY', 'DEBORAH', 'JESSICA', 'SHIRLEY', 'CYNTHIA', 'ANGELA', 'MELISSA',
            'BRENDA', 'ANNA', 'VIRGINIA', 'KATHLEEN', 'PAMELA', 'MARTHA', 'DEBRA',
            'AMANDA', 'STEPHANIE', 'CAROLYN', 'CHRISTINE', 'MARIE', 'JANET', 'CATHERINE',
            'FRANCES', 'CHRISTINA', 'JOAN', 'EVELYN', 'ALICE', 'JULIE', 'JOYCE', 'DIANE',
            'GLORIA', 'JEAN', 'TERESA', 'DORIS', 'JUDITH', 'ROSE', 'JANICE', 'KELLY',
            'NICOLE', 'JUDY', 'CHERYL', 'KATHY', 'THERESA', 'BEVERLY', 'DENISE', 'TAMMY',
            'IRENE', 'JANE', 'LORI', 'MARILYN', 'KATHRYN', 'LOUISE', 'SARA', 'ANNE',
            'JACQUELINE', 'WANDA', 'BONNIE', 'JULIA', 'RUBY', 'LOIS', 'TINA', 'PHYLLIS',
            'NORMA', 'PAULA', 'DIANA', 'ANNIE', 'LILLIAN', 'EMILY', 'ROBIN', 'PEGGY',
            'CRYSTAL', 'GLADYS', 'RITA', 'DAWN', 'CONNIE', 'FLORENCE', 'TRACY', 'EDNA',
            'TIFFANY', 'CARMEN', 'ROSA', 'CINDY', 'GRACE', 'WENDY', 'VICTORIA', 'EDITH',
            'KIM', 'SHERRY', 'SYLVIA', 'JOSEPHINE', 'THELMA', 'SHANNON', 'SHEILA', 'ETHEL',
            'ELLEN', 'ELAINE', 'MARJORIE', 'CARRIE', 'CHARLOTTE', 'MONICA', 'ESTHER',
            'PAULINE', 'EMMA', 'JUANITA', 'ANITA', 'RHONDA', 'HAZEL', 'AMBER', 'EVA',
            'APRIL', 'LESLIE', 'CLARA', 'LUCILLE', 'JAMIE', 'JOANNE', 'ELEANOR', 'VALERIE',
            'DANIELLE', 'MEGAN', 'ALICIA', 'SUZANNE', 'MICHELE', 'GAIL', 'BERTHA', 'DARLENE',
            'VERONICA', 'JILL', 'ERIN', 'GERALDINE', 'LAUREN', 'CATHY', 'JOANN', 'LORRAINE',
            'LYNN', 'SALLY', 'REGINA', 'ERICA', 'BEATRICE', 'DOLORES', 'BERNICE', 'AUDREY',
            'YVONNE', 'ANNETTE', 'SAMANTHA', 'MARION', 'DANA', 'STACY', 'ANA', 'RENEE',
            'IDA', 'VIVIAN', 'ROBERTA', 'HOLLY', 'BRITTANY', 'MELANIE', 'LORETTA', 'YOLANDA',
            'JEANETTE', 'LAURIE', 'KATIE', 'KRISTEN', 'VANESSA', 'ALMA', 'SUE', 'ELSIE',
            'BETH', 'MILDRED', 'ALLISON', 'TAMARA', 'SANDY', 'BELINDA', 'COLLEEN', 'BILLIE',
            
            # Additional unisex/shorter forms
            'JORDAN', 'TAYLOR', 'MORGAN', 'CASEY', 'JAMIE', 'ALEX', 'CHRIS', 'PAT',
            'SAM', 'ROBIN', 'TERRY', 'LEE', 'JEAN', 'DANA', 'LYNN', 'SHANNON', 'KIM',
            'TRACY', 'JESSE', 'CAMERON', 'BLAKE', 'DREW', 'RILEY'
        ])
        
        # Common last names - focused on established families
        self.common_last_names = set([
            'SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN', 'JONES', 'GARCIA', 'MILLER', 'DAVIS',
            'RODRIGUEZ', 'MARTINEZ', 'HERNANDEZ', 'LOPEZ', 'GONZALEZ', 'WILSON', 'ANDERSON',
            'THOMAS', 'TAYLOR', 'MOORE', 'JACKSON', 'MARTIN', 'LEE', 'PEREZ', 'THOMPSON',
            'WHITE', 'HARRIS', 'SANCHEZ', 'CLARK', 'RAMIREZ', 'LEWIS', 'ROBINSON', 'WALKER',
            'YOUNG', 'ALLEN', 'KING', 'WRIGHT', 'SCOTT', 'TORRES', 'NGUYEN', 'HILL', 'FLORES',
            'GREEN', 'ADAMS', 'NELSON', 'BAKER', 'HALL', 'RIVERA', 'CAMPBELL', 'MITCHELL',
            'CARTER', 'ROBERTS', 'GOMEZ', 'PHILLIPS', 'EVANS', 'TURNER', 'DIAZ', 'PARKER',
            'CRUZ', 'EDWARDS', 'COLLINS', 'REYES', 'STEWART', 'MORRIS', 'MORALES', 'MURPHY',
            'COOK', 'ROGERS', 'GUTIERREZ', 'ORTIZ', 'MORGAN', 'COOPER', 'PETERSON', 'BAILEY',
            'REED', 'KELLY', 'HOWARD', 'RAMOS', 'KIM', 'COX', 'WARD', 'RICHARDSON', 'WATSON',
            'BROOKS', 'CHAVEZ', 'WOOD', 'JAMES', 'BENNETT', 'GRAY', 'MENDOZA', 'RUIZ',
            'HUGHES', 'PRICE', 'ALVAREZ', 'CASTILLO', 'SANDERS', 'PATEL', 'MYERS', 'LONG',
            'ROSS', 'FOSTER', 'JIMENEZ', 'POWELL', 'JENKINS', 'PERRY', 'RUSSELL', 'SULLIVAN',
            'BELL', 'COLEMAN', 'BUTLER', 'HENDERSON', 'BARNES', 'GONZALES', 'FISHER', 'VASQUEZ',
            'SIMMONS', 'ROMERO', 'JORDAN', 'PATTERSON', 'ALEXANDER', 'HAMILTON', 'GRAHAM',
            'REYNOLDS', 'GRIFFIN', 'WALLACE', 'MORENO', 'WEST', 'COLE', 'HAYES', 'BRYANT',
            'HERRERA', 'GIBSON', 'ELLIS', 'TRAN', 'MEDINA', 'AGUILAR', 'STEVENS', 'MURRAY',
            'FORD', 'CASTRO', 'MARSHALL', 'OWENS', 'HARRISON', 'FERNANDEZ', 'MCDONALD',
            'WOODS', 'WASHINGTON', 'KENNEDY', 'WELLS', 'VARGAS', 'HENRY', 'CHEN', 'FREEMAN',
            'WEBB', 'TUCKER', 'GUZMAN', 'BURNS', 'CRAWFORD', 'OLSON', 'SIMPSON', 'PORTER',
            'HUNTER', 'GORDON', 'MENDEZ', 'SILVA', 'SHAW', 'SNYDER', 'MASON', 'DIXON',
            'MUNOZ', 'HUNT', 'HICKS', 'HOLMES', 'PALMER', 'WAGNER', 'BLACK', 'ROBERTSON',
            'BOYD', 'ROSE', 'STONE', 'SALAZAR', 'FOX', 'WARREN', 'MILLS', 'MEYER', 'RICE',
            'SCHMIDT', 'GARZA', 'DANIELS', 'FERGUSON', 'NICHOLS', 'STEPHENS', 'SOTO', 'WEAVER',
            'RYAN', 'GARDNER', 'PAYNE', 'GRANT', 'DUNN', 'KELLEY', 'SPENCER', 'HAWKINS',
            'ARNOLD', 'PIERCE', 'VAZQUEZ', 'HANSEN', 'PETERS', 'SANTOS', 'HART', 'BRADLEY',
            'KNIGHT', 'ELLIOTT', 'CUNNINGHAM', 'DUNCAN', 'ARMSTRONG', 'HUDSON', 'CARROLL',
            'LANE', 'RILEY', 'ANDREWS', 'ALVARADO', 'RAY', 'DELGADO', 'BERRY', 'PERKINS',
            'HOFFMAN', 'JOHNSTON', 'MATTHEWS', 'PENA', 'RICHARDS', 'CONTRERAS', 'WILLIS',
            'CARPENTER', 'LAWRENCE', 'SANDOVAL', 'GUERRERO', 'GEORGE', 'CHAPMAN', 'RIOS',
            'ESTRADA', 'ORTEGA', 'WATKINS', 'GREENE', 'NUNEZ', 'WHEELER', 'VALDEZ', 'HARPER',
            'BURKE', 'LARSON', 'SANTIAGO', 'MALDONADO', 'MORRISON', 'FRANKLIN', 'CARLSON',
            'AUSTIN', 'DOMINGUEZ', 'CARR', 'LAWSON', 'JACOBS', 'OBRIEN', 'LYNCH', 'SINGH',
            'VEGA', 'BISHOP', 'MONTGOMERY', 'OLIVER', 'JENSEN', 'HARVEY', 'WILLIAMSON',
            'GILBERT', 'DEAN', 'SIMS', 'ESPINOZA', 'HOWELL', 'LI', 'WONG', 'REID', 'HANSON',
            'LE', 'MCCOY', 'GARRETT', 'BURTON', 'FULLER', 'WANG', 'WEBER', 'WELCH', 'ROJAS',
            'LUCAS', 'MARQUEZ', 'FIELDS', 'PARK', 'YANG', 'LITTLE', 'BANKS', 'PADILLA', 'DAY',
            'WALSH', 'BOWMAN', 'SCHULTZ', 'LUNA', 'FOWLER', 'MEJIA', 'DAVIDSON', 'ACOSTA',
            'BREWER', 'MAY', 'HOLLAND', 'JUAREZ', 'NEWMAN', 'PEARSON', 'CURTIS', 'CORTEZ',
            'DOUGLAS', 'SCHNEIDER', 'JOSEPH', 'BARRETT', 'NAVARRO', 'FIGUEROA', 'KELLER',
            'AVILA', 'WADE', 'MOLINA', 'STANLEY', 'HOPKINS', 'CAMPOS', 'BARNETT', 'BATES',
            'CHAMBERS', 'CALDWELL', 'BECK', 'LAMBERT', 'MIRANDA', 'BYRD', 'CRAIG', 'AYALA',
            'LOWE', 'FRAZIER', 'POWERS', 'NEAL', 'LEONARD', 'GREGORY', 'CARRILLO', 'SUTTON',
            'FLEMING', 'RHODES', 'SHELTON', 'SCHWARTZ', 'NORRIS', 'JENNINGS', 'WATTS',
            'DURAN', 'WALTERS', 'COHEN', 'MCDANIEL', 'MORAN', 'PARKS', 'STEELE', 'VAUGHN',
            'BECKER', 'HOLT', 'DELEON', 'BARKER', 'TERRY', 'HALE', 'LEON', 'HAIL', 'BENSON',
            'HAYNES', 'HORTON', 'MILES', 'LYONS', 'PHAM', 'GRAVES', 'BUSH', 'THORNTON',
            'WOLFE', 'WARNER', 'CABRERA', 'MCKINNEY', 'MANN', 'ZIMMERMAN', 'DAWSON', 'LARA',
            'FLETCHER', 'PAGE', 'MCCARTHY', 'LOVE', 'ROBLES', 'CERVANTES', 'SOLIS', 'ERICKSON',
            'REEVES', 'CHANG', 'KLEIN', 'SALINAS', 'FUENTES', 'BALDWIN', 'DANIEL', 'SIMON',
            'VELASQUEZ', 'HARDY', 'HIGGINS', 'AGUIRRE', 'LIN', 'CUMMINGS', 'CHANDLER',
            'SHARP', 'BARBER', 'BOWEN', 'OCHOA', 'DENNIS', 'ROBBINS', 'LIU', 'RAMSEY',
            'FRANCIS', 'GRIFFITH', 'PAUL', 'BLAIR', 'OCONNOR', 'CARDENAS', 'PACHECO',
            'CROSS', 'CALDERON', 'QUINN', 'MOSS', 'SWANSON', 'CHAN', 'RIVAS', 'KHAN',
            'RODGERS', 'SERRANO', 'FITZGERALD', 'ROSALES', 'STEVENSON', 'CHRISTENSEN',
            'MANNING', 'GILL', 'CURRY', 'MCLAUGHLIN', 'HARMON', 'MCGEE', 'GROSS', 'DOYLE',
            'GARNER', 'NEWTON', 'BURGESS', 'REESE', 'WALTON', 'BLAKE', 'TRUJILLO', 'ADKINS',
            'BRADY', 'GOODMAN', 'ROMAN', 'WEBSTER', 'GOODWIN', 'FISCHER', 'HUANG', 'POTTER',
            'DELACRUZ', 'MONTOYA', 'TODD', 'WU', 'HINES', 'MULLINS', 'CASTANEDA', 'MALONE',
            'CANNON', 'TATE', 'MACK', 'SHERMAN', 'HUBBARD', 'HODGES', 'ZHANG', 'GUERRA',
            'WOLF', 'VALENCIA', 'FRANCO', 'SAUNDERS', 'ROWE', 'GALLAGHER', 'FARMER',
            'HAMMOND', 'HAMPTON', 'TOWNSEND', 'INGRAM', 'WISE', 'GALLEGOS', 'CLARKE',
            'BARTON', 'SCHROEDER', 'MAXWELL', 'WATERS', 'LOGAN', 'CAMACHO', 'STRICKLAND',
            'NORMAN', 'PERSON', 'COLON', 'PARSONS', 'FRANK', 'HARRINGTON', 'GLOVER',
            'OSBORNE', 'BUCHANAN', 'CASEY', 'FLOYD', 'PATTON', 'IBARRA', 'BALL', 'TYLER',
            'SUAREZ', 'BOWERS', 'OROZCO', 'SALAS', 'COBB', 'GIBBS', 'ANDRADE', 'BAUER',
            'CONNER', 'MOODY', 'ESCOBAR', 'MCGUIRE', 'LLOYD', 'MUELLER', 'HARTMAN',
            'FRENCH', 'KRAMER', 'MCBRIDE', 'POPE', 'LINDSEY', 'VELAZQUEZ', 'NORTON',
            'MCCORMICK', 'SPARKS', 'FLYNN', 'YATES', 'HOGAN', 'MARSH', 'MACIAS', 'VILLANUEVA',
            'ZAMORA', 'PRATT', 'STOKES', 'OWEN', 'BALLARD', 'LANG', 'BROCK', 'VILLARREAL',
            'CHARLES', 'DRAKE', 'BARRERA', 'CAIN', 'PATRICK', 'PINEDA', 'BURNETT', 'MERCADO',
            'SANTANA', 'SHEPHERD', 'BAUTISTA', 'ALI', 'SHAFFER', 'LAMB', 'TREVINO', 'MCKENZIE',
            'HESS', 'BEIL', 'OLSEN', 'COCHRAN', 'MORTON', 'NASH', 'WILKINS', 'PETERSEN',
            'BRIGGS', 'SHAH', 'ROTH', 'NICHOLSON', 'HOLLOWAY', 'LOZANO', 'FLOWERS',
            'RANGEL', 'HOOVER', 'ARIAS', 'SHORT', 'MORA', 'VALENZUELA', 'BRYAN', 'MEYERS',
            'WEISS', 'UNDERWOOD', 'BASS', 'GREER', 'SUMMERS', 'HOUSTON', 'CARSON', 'MORROW',
            'CLAYTON', 'WHITAKER', 'DECKER', 'YODER', 'COLLIER', 'ZUNIGA', 'CAREY', 'WILCOX',
            'MELENDEZ', 'POOLE', 'ROBERSON', 'LARSEN', 'CONLEY', 'DAVENPORT', 'COPELAND',
            'MASSEY', 'LAM', 'HUFF', 'ROCHA', 'CAMERON', 'JEFFERSON', 'HOOD', 'MONROE',
            'ANTHONY', 'PITTMAN', 'HUYNH', 'RANDALL', 'SINGLETON', 'KIRK', 'COMBS', 'MATHIS',
            'CHRISTIAN', 'SKINNER', 'BRADFORD', 'RICHARD', 'GALVAN', 'WALL', 'BOONE',
            'KIRBY', 'WILKINSON', 'BRIDGES', 'BRUCE', 'ATKINSON', 'CANTRELL', 'REYES',
            'GUNTER', 'TRAN', 'TYLER', 'HANCOCK', 'AGUIRRE', 'MUELLER', 'HERMAN'
        ])
    
    def preprocess_name(self, name: str) -> str:
        """Clean and standardize name before classification."""
        if not name:
            return ""
        
        # Convert to uppercase
        name = str(name).upper().strip()
        
        # Fix common truncations
        truncation_fixes = {
            r'HUMAN SERVIC$': 'HUMAN SERVICES',
            r'COMM DEVELOP$': 'COMMUNITY DEVELOPMENT',
            r'ASSOC$': 'ASSOCIATION',
            r'CORPORAT$': 'CORPORATION',
            r'APARTMEN$': 'APARTMENTS',
            r'PROPERT$': 'PROPERTY',
            r'INVESTMEN$': 'INVESTMENT',
            r'MANAGEMEN$': 'MANAGEMENT',
            r'RESIDENTIA$': 'RESIDENTIAL',
            r'DEVELOP$': 'DEVELOPMENT',
            r'COMMUNIT$': 'COMMUNITY',
            r'AUTHOR$': 'AUTHORITY',
            r'DIST$': 'DISTRICT',
            r'EDUC$': 'EDUCATION',
            r'SERV$': 'SERVICES',
            r'DEPT$': 'DEPARTMENT',
            r'BLDG$': 'BUILDING',
            r'ASSN$': 'ASSOCIATION',
            r'CTR$': 'CENTER',
            r'CRED$': 'CREDIT',
            r'BAPT$': 'BAPTIST',
            r'PENTECOSTES$': 'PENTECOSTES',
            r'ADVENTIS$': 'ADVENTIST',
            r'METHODIS$': 'METHODIST',
            r'PRESBYTERIA$': 'PRESBYTERIAN',
            r'HEARTL$': 'HEARTLAND',
            r'SERVIC$': 'SERVICES',
            r'PUERTA$': 'PUERTA',
            r'COMM$': 'COMMUNITY',
            r'NATIONA$': 'NATIONAL',
            r'INTERNATIONA$': 'INTERNATIONAL',
            r'CHRISTIA$': 'CHRISTIAN',
            r'FOUNDATIO$': 'FOUNDATION',
            r'AMERICA$': 'AMERICAN',
            r'AMER$': 'AMERICA',
            r'DEPART$': 'DEPARTMENT',
            r'SCHO$': 'SCHOOL',
            r'EDUCATIO$': 'EDUCATION',
            r'REHABILITATIO$': 'REHABILITATION',
            r'ASSOCIATIO$': 'ASSOCIATION',
            r'ORGANIZATIO$': 'ORGANIZATION',
            r'ADMINISTRATO$': 'ADMINISTRATOR',
            r'COMMUNICATIO$': 'COMMUNICATION',
            r'TRANSPORTATIO$': 'TRANSPORTATION',
            r'INFORMATIO$': 'INFORMATION',
            r'CONSTRUCTIO$': 'CONSTRUCTION',
            r'DISTRIBUTIO$': 'DISTRIBUTION',
            r'PRODUCTIO$': 'PRODUCTION',
            r'OPERATIO$': 'OPERATION',
            r'RECREATIO$': 'RECREATION',
            r'EDUCATIO$': 'EDUCATION',
            r'FOUNDATIO$': 'FOUNDATION',
            r'CORPORATIO$': 'CORPORATION',
            r'ASSOCIATIO$': 'ASSOCIATION',
            r'FEDERATIO$': 'FEDERATION',
            r'ORGANIZATIO$': 'ORGANIZATION',
            r'INSTITUTIO$': 'INSTITUTION',
            r'COMMISSIO$': 'COMMISSION',
            r'COALITIO$': 'COALITION',
            r'ALLIANC$': 'ALLIANCE',
            r'PARTNERSHI$': 'PARTNERSHIP',
            r'COOPERATIV$': 'COOPERATIVE',
            r'COLLABORATIV$': 'COLLABORATIVE',
            r'INITIATIV$': 'INITIATIVE',
            r'ENTERPRIS$': 'ENTERPRISE',
            r'VENTUR$': 'VENTURE',
            r'PROJEC$': 'PROJECT',
            r'PROGRA$': 'PROGRAM',
            r'SERVIC$': 'SERVICE',
            r'SYSTE$': 'SYSTEM',
            r'NETWOR$': 'NETWORK',
            r'CENTE$': 'CENTER',
            r'OFFIC$': 'OFFICE',
            r'BUREA$': 'BUREAU',
            r'AGENC$': 'AGENCY',
            r'DIVISIO$': 'DIVISION',
            r'BRANC$': 'BRANCH',
            r'SECTIO$': 'SECTION',
            r'DEPARTMEN$': 'DEPARTMENT',
            r'ADMINISTRATI$': 'ADMINISTRATION',
            r'MANAGEMEN$': 'MANAGEMENT',
            r'OPERATIO$': 'OPERATION',
            r'MAINTENANC$': 'MAINTENANCE',
            r'PRESERVATIO$': 'PRESERVATION',
            r'PROTECTIO$': 'PROTECTION',
            r'PREVENTIO$': 'PREVENTION',
            r'INTERVENTIO$': 'INTERVENTION',
            r'REHABILITAT$': 'REHABILITATION',
            r'RESTORATI$': 'RESTORATION',
            r'RENOVATI$': 'RENOVATION',
            r'RECONSTRUCTI$': 'RECONSTRUCTION',
            r'REDEVELOPMEN$': 'REDEVELOPMENT',
            r'REVITALIZATI$': 'REVITALIZATION',
        }
        
        for pattern, replacement in truncation_fixes.items():
            name = re.sub(pattern, replacement, name)
        
        # Fix ETA -> ET AL
        name = re.sub(r'\bETA\b', 'ET AL', name)
        
        # Fix typos
        name = name.replace('COPERATION', 'CORPORATION')
        name = name.replace('CHALLEG', 'CHALLENGE')
        name = name.replace('PENTECOSTES', 'PENTECOSTAL')
        name = name.replace('NEBRSAKA', 'NEBRASKA')
        
        # Standardize spacing around punctuation
        name = re.sub(r'\s*&\s*', ' & ', name)
        name = re.sub(r'\s+-\s+', '-', name)
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip()
    
    def has_hyphenated_name_pattern(self, name: str) -> bool:
        """Check if name has hyphenated surname pattern (common in Hispanic names)."""
        # Pattern for Hispanic double surnames or hyphenated names
        if re.search(r'[A-Z][a-z]+-[A-Z][a-z]+', name):
            return True
        # Check if hyphenated part appears before common first names
        parts = name.split()
        if len(parts) >= 2 and '-' in parts[0]:
            # Check if later parts are common first names
            for part in parts[1:]:
                if part.upper() in self.common_first_names:
                    return True
        return False
    
    def extract_entity_type_ignoring_numbers(self, name: str) -> str:
        """Extract core entity type, ignoring numbers."""
        # Remove numbers but preserve entity markers
        cleaned = re.sub(r'\d+', '', name)
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def calculate_confidence(self, name: str, entity_type: str, indicators: List[str]) -> float:
        """Calculate confidence score based on multiple factors."""
        base_score = 0.1  # Start with low base
        
        # Strong indicators for each type
        strong_patterns = {
            'LLC': [r'\bLLC\b', r'\bL\.L\.C\.', r'\bLIMITED LIABILITY'],
            'Corporation': [r'\bINC\b', r'\bCORP\b', r'\bINCORPORATED\b', r'\bCORPORATION\b'],
            'Trust': [r'\bTRUST\b', r'\bTRUSTEE\b', r'\bLIVING TRUST\b', r'\bREVOCABLE\b'],
            'Government': [r'\bCITY OF\b', r'\bCOUNTY OF\b', r'\bSTATE OF\b', r'\bAUTHORITY\b'],
            'Religious_Nonprofit': [r'\bCHURCH\b', r'\bTEMPLE\b', r'\bMOSQUE\b', r'\bPARISH\b'],
            'Estate': [r'\bESTATE OF\b', r'\bESTATE\b', r'\bDECEASED\b'],
            'Healthcare': [r'\bHOSPITAL\b', r'\bMEDICAL\b', r'\bCLINIC\b'],
            'Bank': [r'\bBANK\b', r'\bCREDIT UNION\b'],
            'Condo': [r'\bCONDOMINIUM\b', r'\bCONDO\b', r'\bHOA\b', r'\bHOMEOWNERS\b'],
        }
        
        # Check for strong indicators
        for pattern in strong_patterns.get(entity_type, []):
            if re.search(pattern, name):
                base_score += 0.4
                break  # Only count once
        
        # Check for strong trust indicators specifically
        if entity_type == 'Trust':
            for pattern in self.strong_trust_indicators:
                if re.search(pattern, name):
                    base_score += 0.3
                    break
        
        # Bonus for multiple corroborating indicators
        if len(indicators) > 2:
            base_score += 0.1 * min(3, len(indicators) - 2)  # Cap bonus at 0.3
        
        # Penalty for very short names (likely truncated)
        if len(name) < 10:
            base_score -= 0.1
        
        # Penalty for ending in truncation pattern
        for pattern in self.truncated_org_patterns:
            if re.search(pattern, name):
                base_score -= 0.1
                break
        
        # Individual-specific scoring
        if entity_type == 'Individual':
            # Check for personal name patterns
            parts = name.split()
            first_name_match = any(part in self.common_first_names for part in parts)
            last_name_match = any(part in self.common_last_names for part in parts)
            
            if first_name_match and last_name_match:
                base_score += 0.5
            elif first_name_match or last_name_match:
                base_score += 0.2
            
            # Check for personal indicators (Jr, Sr, etc.)
            for pattern in self.personal_patterns[:8]:  # First 8 are strongest
                if re.search(pattern, name):
                    base_score += 0.2
                    break
            
            # Penalty if has business indicators
            if any(re.search(p, name) for p in self.business_patterns[:10]):
                base_score -= 0.3
        
        # Corporate/business entity bonus
        elif entity_type in ['Corporation', 'LLC', 'Corporate_Other']:
            # Multiple business patterns increase confidence
            business_count = sum(1 for p in self.business_patterns if re.search(p, name))
            if business_count > 0:
                base_score += min(0.3, business_count * 0.1)
        
        return min(1.0, max(0.0, base_score))
    
    def classify_owner(self, name: str) -> Tuple[str, float, List[str]]:
        """
        Classify owner name with confidence score.
        Returns: (entity_type, confidence, indicators_found)
        """
        if pd.isna(name) or name == "":
            return "Unknown", 0.0, []
        
        # Preprocess name
        name = self.preprocess_name(name)
        upper_name = name.upper()
        
        # For number extraction, clean the name
        name_without_numbers = self.extract_entity_type_ignoring_numbers(upper_name)
        
        indicators = []
        
        # Check for hyphenated Hispanic names
        if self.has_hyphenated_name_pattern(name):
            indicators.append("hyphenated_hispanic_name")
        
        # Check Government patterns with priority
        for pattern in self.gov_patterns:
            if re.search(pattern, upper_name):
                indicators.append("gov_pattern")
                # Strong government indicators
                if any(re.search(p, upper_name) for p in [r'\bCITY OF\b', r'\bCOUNTY OF\b', 
                                                            r'\bSTATE OF\b', r'\bAUTHORITY\b',
                                                            r'\bSCHOOL DISTRICT\b']):
                    confidence = self.calculate_confidence(name, 'Government', indicators)
                    return "Government", max(0.7, confidence), indicators
                break
        
        # Check Religious/Non-profit patterns
        for pattern in self.religious_nonprofit_patterns:
            if re.search(pattern, upper_name):
                indicators.append("religious_nonprofit")
                confidence = self.calculate_confidence(name, 'Religious_Nonprofit', indicators)
                return "Religious_Nonprofit", confidence, indicators
        
        # Check Healthcare patterns
        for pattern in self.healthcare_patterns:
            if re.search(pattern, upper_name):
                indicators.append("healthcare")
                confidence = self.calculate_confidence(name, 'Healthcare', indicators)
                return "Healthcare", confidence, indicators
        
        # Check Bank patterns
        for pattern in self.bank_patterns:
            if re.search(pattern, upper_name):
                indicators.append("bank")
                confidence = self.calculate_confidence(name, 'Bank', indicators)
                return "Bank", confidence, indicators
        
        # Check Condo/HOA patterns
        for pattern in self.condo_patterns:
            if re.search(pattern, upper_name):
                indicators.append("condo")
                confidence = self.calculate_confidence(name, 'Condo', indicators)
                return "Condo", confidence, indicators
        
        # Check Housing patterns
        for pattern in self.housing_patterns:
            if re.search(pattern, upper_name):
                indicators.append("housing")
                # Could be government or non-profit
                if "AUTHORITY" in upper_name or "HOUSING AUTH" in upper_name:
                    return "Government", 0.7, indicators
                else:
                    return "Corporate_Other", 0.5, indicators
        
        # Check LLC patterns (use cleaned name for better matching)
        for pattern in self.llc_patterns:
            if re.search(pattern, name_without_numbers):
                indicators.append("llc_pattern")
                confidence = self.calculate_confidence(name, 'LLC', indicators)
                return "LLC", confidence, indicators
        
        # Check Corporation patterns
        for pattern in self.corp_suffixes:
            if re.search(pattern, name_without_numbers):
                indicators.append("corp_suffix")
                # Additional check for "has_business_suffix"
                if re.search(r'\b(?:INC|CORP|CORPORATION|COMPANY|CO|LTD|LIMITED)\.?(?:\s|$)', upper_name):
                    indicators.append("has_business_suffix")
                confidence = self.calculate_confidence(name, 'Corporation', indicators)
                return "Corporation", confidence, indicators
        
        # Check Trust patterns
        for pattern in self.trust_patterns:
            if re.search(pattern, upper_name):
                indicators.append("trust_pattern")
                confidence = self.calculate_confidence(name, 'Trust', indicators)
                return "Trust", confidence, indicators
        
        # Check Estate patterns
        for pattern in self.estate_patterns:
            if re.search(pattern, upper_name):
                indicators.append("estate_pattern")
                # Check for "REAL ESTATE" which should be corporate
                if re.search(r'\bREAL ESTATE\b', upper_name):
                    for pattern in self.business_patterns:
                        if re.search(pattern, upper_name):
                            indicators.append("business_terms_2")
                            break
                    return "Corporate_Other", 0.4, indicators
                confidence = self.calculate_confidence(name, 'Estate', indicators)
                return "Estate", confidence, indicators
        
        # Check for business patterns (multiple levels)
        business_score = 0
        business_patterns_found = []
        
        # Check primary business patterns
        primary_business = [r'\bPROPERTIES\b', r'\bHOLDINGS\b', r'\bGROUP\b', r'\bPARTNERS\b',
                           r'\bVENTURES\b', r'\bENTERPRISES\b', r'\bDEVELOPMENT\b', 
                           r'\bMANAGEMENT\b', r'\bINVESTMENTS?\b', r'\bCAPITAL\b',
                           r'\bREALTY\b', r'\bREAL ESTATE\b']
        
        for pattern in primary_business:
            if re.search(pattern, upper_name):
                business_score += 1
                business_patterns_found.append("business_terms_1")
                break
        
        # Check secondary business patterns
        for pattern in self.business_patterns:
            if re.search(pattern, upper_name):
                business_score += 0.5
                if "business_terms_2" not in business_patterns_found:
                    business_patterns_found.append("business_terms_2")
                break
        
        # International name patterns
        for pattern_type, pattern in self.international_patterns.items():
            if re.search(pattern, upper_name):
                indicators.append(f"international_{pattern_type}")
        
        # Check if likely individual based on patterns
        parts = upper_name.split()
        
        # Count personal indicators
        personal_score = 0
        
        # Check for common first/last name combinations
        if len(parts) >= 2:
            # Check standard "FIRST LAST" pattern
            if parts[0] in self.common_first_names and parts[-1] in self.common_last_names:
                personal_score += 3
                indicators.append("common_firstname_first")
                indicators.append("common_lastname_last")
            # Check "LAST, FIRST" pattern
            elif parts[0].rstrip(',') in self.common_last_names and len(parts) > 1 and parts[1] in self.common_first_names:
                personal_score += 3
                indicators.append("common_lastname_first")
                indicators.append("common_firstname_second")
            # Check if any part is a common first name
            elif any(part in self.common_first_names for part in parts):
                personal_score += 1
                indicators.append("has_common_firstname")
            # Check if any part is a common last name
            elif any(part in self.common_last_names for part in parts):
                personal_score += 1
                indicators.append("has_common_lastname")
        
        # Check for middle initials (strong individual indicator)
        if re.search(r'\b[A-Z]\.\s*(?:[A-Z]|$)', upper_name) or re.search(r'\b[A-Z]\s+[A-Z]\s+', upper_name):
            personal_score += 1
            indicators.append("has_middle_initial")
        
        # Check for personal patterns (Jr, Sr, etc.)
        for pattern in self.personal_patterns:
            if re.search(pattern, upper_name):
                personal_score += 1.5
                indicators.append("personal_pattern_1")
                break
        
        # Check for conjunctions suggesting multiple people
        if re.search(r'\b(?:AND|&|OR)\b', upper_name):
            indicators.append("has_conjunction")
            # This could be personal or business
            if personal_score > 0:
                personal_score += 0.5
        
        # Check for numbers (often in business names)
        if re.search(r'\d', upper_name):
            indicators.append("contains_numbers")
            business_score += 0.25
        
        # Make classification decision
        if business_score >= 1 and personal_score < 2:
            indicators.extend(business_patterns_found)
            confidence = self.calculate_confidence(name, 'Corporate_Other', indicators)
            return "Corporate_Other", confidence, indicators
        elif personal_score >= 2:
            confidence = self.calculate_confidence(name, 'Individual', indicators)
            return "Individual", confidence, indicators
        elif business_score > 0:
            indicators.extend(business_patterns_found)
            confidence = self.calculate_confidence(name, 'Corporate_Other', indicators)
            return "Corporate_Other", confidence, indicators
        elif personal_score > 0:
            confidence = self.calculate_confidence(name, 'Individual', indicators)
            return "Individual", confidence, indicators
        
        # Check for truncated patterns as last resort
        for pattern in self.truncated_org_patterns:
            if re.search(pattern, upper_name):
                indicators.append("truncated_org")
                return "Corporate_Other", 0.3, indicators
        
        # Default to Unknown
        return "Unknown", 0.0, indicators

# ── Main Analysis Function ─────────────────────────────────────────────
def analyze_owner_classifications(gdb_path: str, parcels_fc: str, output_dir: str):
    """Main analysis function."""
    print("Enhanced Absentee Residential Owner Classification Analysis")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parcels
    print("\nLoading parcels from geodatabase...")
    fields = ["*"]  # Get all fields
    
    try:
        # Read into pandas dataframe
        data = []
        field_names = None
        
        with arcpy.da.SearchCursor(parcels_fc, fields) as cursor:
            field_names = cursor.fields
            for row in cursor:
                data.append(row)
        
        df = pd.DataFrame(data, columns=field_names)
        print(f"  Loaded {len(df):,} parcels")
        
    except Exception as e:
        print(f"Error loading parcels: {e}")
        return
    
    # Identify absentee residential properties
    df = identify_absentee_residential(
        df, 
        PARCEL_ADDR_FIELDS,
        OWNER_ADDR_FIELDS,
        LAND_USE_FIELD,
        RESIDENTIAL_CODES
    )
    
    # Filter to absentee residential only
    absentee_df = df[df['IsAbsenteeResidential']].copy()
    print(f"\nAnalyzing {len(absentee_df):,} absentee residential properties")
    
    if len(absentee_df) == 0:
        print("No absentee residential properties found!")
        return
    
    # Find owner name field
    owner_field = None
    for field in OWNER_NAME_FIELDS:
        if field in absentee_df.columns:
            owner_field = field
            print(f"Using owner name field: {owner_field}")
            break
    
    if not owner_field:
        print("Error: No owner name field found!")
        return
    
    # Initialize classifier
    classifier = EnhancedOwnerNameClassifier()
    
    # Classify owners
    print("\nClassifying owner names...")
    classifications = []
    confidence_scores = []
    indicators_list = []
    
    for idx, row in absentee_df.iterrows():
        owner_name = row[owner_field]
        entity_type, confidence, indicators = classifier.classify_owner(owner_name)
        classifications.append(entity_type)
        confidence_scores.append(confidence)
        indicators_list.append(indicators)
        
        if idx % 5000 == 0:
            print(f"  Processed {idx:,} records...")
    
    absentee_df['ClassifiedType'] = classifications
    absentee_df['ClassificationConfidence'] = confidence_scores
    absentee_df['ClassificationIndicators'] = indicators_list
    
    # Analysis results
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS - ABSENTEE RESIDENTIAL PROPERTIES")
    print("=" * 60)
    
    # Overall statistics
    type_counts = absentee_df['ClassifiedType'].value_counts()
    print("\nEntity Type Distribution:")
    print("-" * 40)
    total = len(absentee_df)
    for entity_type, count in type_counts.items():
        pct = count / total * 100
        print(f"{entity_type:20} {count:8,} ({pct:5.1f}%)")
    
    # Confidence analysis
    print("\nConfidence Score Statistics:")
    print("-" * 40)
    for entity_type in type_counts.index:
        mask = absentee_df['ClassifiedType'] == entity_type
        if mask.any():
            scores = absentee_df.loc[mask, 'ClassificationConfidence']
            print(f"{entity_type:20} "
                  f"Mean: {scores.mean():.3f}, "
                  f"Median: {scores.median():.3f}, "
                  f"Min: {scores.min():.3f}, "
                  f"Max: {scores.max():.3f}")
    
    # Low confidence analysis
    low_conf_threshold = 0.5
    low_conf_mask = absentee_df['ClassificationConfidence'] < low_conf_threshold
    low_conf_count = low_conf_mask.sum()
    print(f"\nLow Confidence Classifications (< {low_conf_threshold}): "
          f"{low_conf_count:,} ({low_conf_count/total*100:.1f}%)")
    
    # Distance analysis if available
    if DISTANCE_FIELD in absentee_df.columns:
        print("\n" + "=" * 60)
        print("DISTANCE ANALYSIS BY ENTITY TYPE")
        print("=" * 60)
        
        # Convert distance to numeric, handling any errors
        absentee_df[DISTANCE_FIELD] = pd.to_numeric(absentee_df[DISTANCE_FIELD], errors='coerce')
        
        # Remove invalid distances
        valid_dist_mask = absentee_df[DISTANCE_FIELD].notna() & (absentee_df[DISTANCE_FIELD] >= 0)
        dist_df = absentee_df[valid_dist_mask].copy()
        
        if len(dist_df) > 0:
            print(f"\nAnalyzing {len(dist_df):,} properties with valid distances")
            print("-" * 60)
            
            for entity_type in type_counts.index:
                mask = dist_df['ClassifiedType'] == entity_type
                if mask.any():
                    distances = dist_df.loc[mask, DISTANCE_FIELD]
                    if len(distances) > 0:
                        print(f"\n{entity_type}:")
                        print(f"  Count: {len(distances):,}")
                        print(f"  Mean Distance: {distances.mean():.1f} km")
                        print(f"  Median Distance: {distances.median():.1f} km")
                        print(f"  25th Percentile: {distances.quantile(0.25):.1f} km")
                        print(f"  75th Percentile: {distances.quantile(0.75):.1f} km")
                        print(f"  Max Distance: {distances.max():.1f} km")
                        
                        # Local vs distant
                        local = (distances < 50).sum()
                        distant = (distances >= 50).sum()
                        if len(distances) > 0:
                            print(f"  Local (<50km): {local:,} ({local/len(distances)*100:.1f}%)")
                            print(f"  Distant (≥50km): {distant:,} ({distant/len(distances)*100:.1f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Entity type distribution
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.title('Absentee Residential Property Owner Types')
    plt.xlabel('Entity Type')
    plt.ylabel('Number of Properties')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_type_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Confidence score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(absentee_df['ClassificationConfidence'], bins=50, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Low Confidence Threshold')
    plt.title('Classification Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Properties')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Distance boxplot by entity type (if available)
    if DISTANCE_FIELD in absentee_df.columns and len(dist_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        plot_data = []
        plot_labels = []
        
        for entity_type in type_counts.index:
            mask = dist_df['ClassifiedType'] == entity_type
            if mask.any():
                distances = dist_df.loc[mask, DISTANCE_FIELD].dropna()
                if len(distances) > 0:
                    plot_data.append(distances)
                    plot_labels.append(f"{entity_type}\n(n={len(distances):,})")
        
        if plot_data:
            plt.boxplot(plot_data, labels=plot_labels)
            plt.title('Owner Distance Distribution by Entity Type')
            plt.ylabel('Distance (km)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'distance_by_entity_type.png'), dpi=300)
            plt.close()
    
    # Save detailed results
    print("\nSaving detailed results...")
    
    # 1. Summary statistics
    summary_file = os.path.join(output_dir, 'classification_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Enhanced Absentee Residential Owner Classification Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Parcels Analyzed: {len(df):,}\n")
        f.write(f"Residential Parcels: {df['IsResidential'].sum():,}\n")
        f.write(f"Owner-Occupied: {df['IsOwnerOccupied'].sum():,}\n")
        f.write(f"Absentee Residential: {len(absentee_df):,}\n\n")
        
        f.write("Entity Type Distribution:\n")
        f.write("-" * 40 + "\n")
        for entity_type, count in type_counts.items():
            pct = count / total * 100
            f.write(f"{entity_type:20} {count:8,} ({pct:5.1f}%)\n")
        
        f.write("\nConfidence Score Statistics:\n")
        f.write("-" * 40 + "\n")
        for entity_type in type_counts.index:
            mask = absentee_df['ClassifiedType'] == entity_type
            if mask.any():
                scores = absentee_df.loc[mask, 'ClassificationConfidence']
                f.write(f"{entity_type:20} "
                       f"Mean: {scores.mean():.3f}, "
                       f"Median: {scores.median():.3f}, "
                       f"Min: {scores.min():.3f}, "
                       f"Max: {scores.max():.3f}\n")
    
    # 2. Sample classifications for each type
    samples_file = os.path.join(output_dir, 'classification_samples.csv')
    sample_data = []
    
    for entity_type in type_counts.index:
        mask = absentee_df['ClassifiedType'] == entity_type
        if mask.any():
            # Get up to 20 samples, prioritizing different confidence levels
            type_df = absentee_df[mask].copy()
            type_df = type_df.sort_values('ClassificationConfidence', ascending=False)
            
            # Take samples from different confidence ranges
            high_conf = type_df[type_df['ClassificationConfidence'] >= 0.7].head(7)
            mid_conf = type_df[(type_df['ClassificationConfidence'] >= 0.4) & 
                              (type_df['ClassificationConfidence'] < 0.7)].head(7)
            low_conf = type_df[type_df['ClassificationConfidence'] < 0.4].head(6)
            
            for df_subset in [high_conf, mid_conf, low_conf]:
                for _, row in df_subset.iterrows():
                    sample_data.append({
                        'EntityType': entity_type,
                        'OwnerName': row[owner_field],
                        'Confidence': row['ClassificationConfidence'],
                        'Indicators': ', '.join(row['ClassificationIndicators']),
                        'Distance_km': row.get(DISTANCE_FIELD, 'N/A')
                    })
    
    pd.DataFrame(sample_data).to_csv(samples_file, index=False)
    
    # 3. Low confidence classifications for review
    low_conf_file = os.path.join(output_dir, 'low_confidence_classifications.txt')
    with open(low_conf_file, 'w') as f:
        f.write("LOW CONFIDENCE CLASSIFICATIONS FOR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file contains owner names with classification confidence < 0.5\n")
        f.write("Format: [Confidence] EntityType | OwnerName | Indicators\n")
        f.write("-" * 80 + "\n\n")
        
        low_conf_df = absentee_df[low_conf_mask].copy()
        low_conf_df = low_conf_df.sort_values(['ClassifiedType', 'ClassificationConfidence'])
        
        current_type = None
        for _, row in low_conf_df.iterrows():
            if current_type != row['ClassifiedType']:
                current_type = row['ClassifiedType']
                count = (low_conf_df['ClassifiedType'] == current_type).sum()
                f.write(f"\n{current_type.upper()} ({count} records)\n")
                f.write("-" * 60 + "\n")
            
            f.write(f"[{row['ClassificationConfidence']:.3f}] "
                   f"{row['ClassifiedType']:15} | "
                   f"{str(row[owner_field])[:50]:50} | "
                   f"{', '.join(row['ClassificationIndicators'])}\n")
    
    # 4. Full classification results
    results_file = os.path.join(output_dir, 'full_classification_results.csv')
    export_cols = [owner_field, 'ClassifiedType', 'ClassificationConfidence', 
                   'IsResidential', 'IsOwnerOccupied', 'IsAbsenteeResidential']
    if DISTANCE_FIELD in absentee_df.columns:
        export_cols.append(DISTANCE_FIELD)
    
    absentee_df[export_cols].to_csv(results_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - Summary: {os.path.basename(summary_file)}")
    print(f"  - Samples: {os.path.basename(samples_file)}")
    print(f"  - Low confidence: {os.path.basename(low_conf_file)}")
    print(f"  - Full results: {os.path.basename(results_file)}")

# ── Main Execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set up ArcGIS environment
    arcpy.env.overwriteOutput = True
    
    # Run analysis
    analyze_owner_classifications(GDB, PARCELS_FC, OUTPUT_DIR)
    match = re.match(pattern1, addr)
    if match:
        components['number'] = match.group(1)
        components['direction'] = match.group(2)[0]  # Just first letter
        components['street'] = match.group(3)
        components['type'] = match.group(4)
        return components
    
    # Pattern 2: Number Street Direction Type (e.g., "123 MAIN ST N")
    pattern2 = r'^(\d+)\s+(.+?)\s+(ST|AVE|RD|DR|LN|BLVD|PL|CT|CIR|TRL|PKWY|HWY|SQ|TER|PLZ|ALY)\s+([NSEW]|NORTH|SOUTH|EAST|WEST)#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
