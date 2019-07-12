from train import xy_gen

path = "../../.."
xlsx_path = "../../../Monash_University_Seizure_Detection_Database_" \
            "September_2018_Deidentified.xlsx"
sheet_name = "Seizure Information"

training_set, validation_set, test_set = xy_gen(path, xlsx_path, sheet_name)


