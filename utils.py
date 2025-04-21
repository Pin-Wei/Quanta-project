#!/usr/bin/python

def standardized_feature_list():
    return [
        "LANGUAGE_ST_SCALED_SIMILARITY", 
        "LANGUAGE_ST_SCALED_VOCABULARY", 
        "LANGUAGE_ST_SCALED_INFORMATION", 
        "MEMORY_ST_SCALED_AudImm", 
        "MEMORY_ST_SCALED_VisImm", 
        "MEMORY_ST_SCALED_WorMem", 
        "MEMORY_ST_SCALED_LogMemI", 
        "MEMORY_ST_SCALED_LetNumSeq", 
        "MEMORY_ST_SCALED_FamPicI", 
        "MEMORY_ST_SCALED_VerPairI", 
        "MEMORY_ST_SCALED_FacI", 
        "MEMORY_ST_SCALED_SpaForward", 
        "MEMORY_ST_SCALED_SpaBackward", 
        "MOTOR_ST_SCALED_FineMotor", 
        "MOTOR_ST_SCALED_Balance", 
        "MOTOR_ST_SCALED_ProcessingSpeed"    
    ]

def domain_approach_mapping_dict():
    return {
        "STRUCTURE": {
            "domains": ["STRUCTURE"],
            "approaches": ["MRI"]
        }, 
        "BEH": {
            "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
            "approaches": ["BEH"]
        },
        "FUNCTIONAL": {
            "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
            "approaches": ["EEG", "MRI"]
        }, 
        "ALL": {
            "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
            "approaches": ["MRI", "BEH", "EEG"]
        }
    }