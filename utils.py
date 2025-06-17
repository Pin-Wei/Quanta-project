#!/usr/bin/python

def basic_Q_features(): # from questionnaires
    return [
        "BASIC_Q_EHI_Sum", 
        "BASIC_Q_SF36_PhysicalFunct", 
        "BASIC_Q_SF36_PhysicalLimit", 
        "BASIC_Q_SF36_EmotionalWell", 
        "BASIC_Q_SF36_EmotionalLimit", 
        "BASIC_Q_SF36_Energy", 
        "BASIC_Q_SF36_SocialFunc", 
        "BASIC_Q_SF36_Pain", 
        "BASIC_Q_SF36_GeneralHealth", 
        "BASIC_Q_SF36_Physical", 
        "BASIC_Q_SF36_Mental", 
        "BASIC_Q_PSQI_SleepQuality", 
        "BASIC_Q_PSQI_SleepLatency", 
        "BASIC_Q_PSQI_SleepDuration", 
        "BASIC_Q_PSQI_SleepEfficiency", 
        "BASIC_Q_PSQI_SleepDisturbance", 
        "BASIC_Q_PSQI_SleepMedication", 
        "BASIC_Q_PSQI_DaytimeDysfunc", 
        "BASIC_Q_PSQI_Sum", 
        "BASIC_Q_IPAQ_MET", 
        "BASIC_Q_BFI_Extraversion", 
        "BASIC_Q_BFI_Agreeableness", 
        "BASIC_Q_BFI_Conscientiousness", 
        "BASIC_Q_BFI_EmotionalStability", 
        "BASIC_Q_BFI_Intellect", 
        "BASIC_Q_MSPSS_Sum", 
        "BASIC_Q_CogFailure_Sum", 
        "BASIC_Q_Beck_Anxiety", 
        "BASIC_Q_Beck_Depression", 
        "BASIC_Q_MOCA_SUM"
    ]

def ST_features(): # from standardized tests
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