import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import joblib
from PIL import Image

# Load your trained model
# Make sure 'solubility_rf_model.pkl' is in the same folder
model = joblib.load("solubility_rf_model.pkl")

def smiles_to_fingerprint(smiles):
    """Convert SMILES to Morgan fingerprint for prediction"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp).reshape(1, -1)

def mol_to_image(smiles):
    """Generate molecule image using RDKit and PIL"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300))
    return img

# Streamlit UI
st.title("Solubility Predictor (logS)")
st.write("Enter a SMILES string to predict aqueous solubility.")

user_input = st.text_input("SMILES", "CCO")

if st.button("Predict"):
    X_new = smiles_to_fingerprint(user_input)
    if X_new is None:
        st.error("Invalid SMILES string!")
    else:
        pred_logS = model.predict(X_new)
        st.success(f"Predicted logS: {pred_logS[0]:.2f}")

        # Display molecule image
        img = mol_to_image(user_input)
        if img:
            st.image(img)
        else:
            st.warning("Cannot generate image for this SMILES.")
