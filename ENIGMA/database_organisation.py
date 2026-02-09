import pandas as pd\

df_mics_data = pd.read_excel("/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/Original-Clinical Database_TA2025.xlsx", sheet_name=1)
df_mics_data = df_mics_data[["participant_id", "MRI.1.scan.AGE", "Gender"]]
df_mics_data["participant_id"] = "sub-" + df_mics_data["participant_id"]
df_mics_data = df_mics_data.rename(columns={"participant_id": "subjectID", "MRI.1.scan.AGE": "Age", "Gender": "Sex"}).dropna()

print(df_mics_data)

df_enigma_data = pd.read_csv("/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList.csv", header=None, names=["subjectID", "path"])
print(df_enigma_data)

df_mics_data = df_mics_data.merge(df_enigma_data, how="inner", on="subjectID").drop_duplicates()
print(df_mics_data)

df_enigma_data = df_mics_data[["subjectID", "path"]]
df_enigma_data.to_csv("/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList_edited.csv", header=False, index=False)

df_mics_data = df_mics_data[["subjectID", "Age", "Sex"]]
df_mics_data['Age'] = df_mics_data['Age'].astype(float)
df_mics_data['Sex'] = df_mics_data['Sex'].replace('F', 1).replace('F ', 1).replace('M', 2)
df_mics_data['Sex'] = df_mics_data['Sex'].astype(int)
df_mics_data.to_csv("/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ALL_Subject_Info.txt", index=False, sep='\t', header=True)

