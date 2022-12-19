# dataset information.
def getInfodataset(idxDataset):
    infoDatasets = {
        "Dataset0": 
            {"nSubjects": 11,
             "idxSubjectsToExclude": [],
             "infoSubjects": {
                   "s0": {"weight": 64.8,"height": 1.76, "originalName": "AdDB", "rotations": {"STS": -90, "Squatjump": -30, "Squat": 150}},
                   "s1": {"weight": 65.5,"height": 1.78, "originalName": "DiVa", "rotations": {"STS": -80, "Squatjump": -30, "Squat": 150}},
                   "s2": {"weight": 71.7,"height": 1.73, "originalName": "ElMa", "rotations": {"STS": -80, "Squatjump": -20, "Squat": -20}},
                   "s3": {"weight": 70.8,"height": 1.80, "originalName": "FaKe", "rotations": {"STS": -90, "Squatjump": -30, "Squat": 150}},
                   "s4": {"weight": 64.2,"height": 1.69, "originalName": "FiBu", "rotations": {"STS": -90, "Squatjump": -30, "Squat": 150}},
                   "s5": {"weight": 60.1,"height": 1.70, "originalName": "FrDG", "rotations": {"STS": -90, "Squatjump": -20, "Squat": 160}},
                   "s6": {"weight": 73.6,"height": 1.75, "originalName": "GeGi", "rotations": {"STS": -80, "Squatjump": -30, "Squat": 160}},
                   "s7": {"weight": 66.3,"height": 1.78, "originalName": "IrSi", "rotations": {"STS": -80, "Squatjump": -20, "Squat": 160}},
                   "s8": {"weight": 81.0,"height": 1.77, "originalName": "JoCR", "rotations": {"STS": -90, "Squatjump": -30, "Squat": 160}},
                   "s9": {"weight": 78.4,"height": 1.90, "originalName": "KeTa", "rotations": {"STS": -90, "Squatjump": -130, "Squat": -130}},
                   "s10": {"weight":73.4,"height": 1.75, "originalName": "TeHo", "rotations": {"STS": -90, "Squatjump": -30, "Squat": 150}}}}}        
    infoDataset = infoDatasets["Dataset" + str(idxDataset)]
    
    return infoDataset