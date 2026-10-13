# Standard imports
import glob
import os
import re

# Third-party imports
import cv2
import numpy as np
import pandas as pd
from pydicom import dcmread


def get_patient_id(patient_csv: pd.core.frame.DataFrame, patient_id: str) -> str:
    """Returns the patient ID from the csv file. This is used to get the contours for the patient.

    Args:
        patient_csv (pd.core.frame.DataFrame): [CSV file containing patient data]
        patient_id (str): [Patient ID]

    Returns:
        str: [Original ID of the patient]
    """
    patient_row = patient_csv.loc[patient_csv['PatientID'] == patient_id]
    return patient_row['OriginalID'].to_string(index=False)


def load_dcm_folder(folder_path: str, verbose: bool = False) -> list:
    """Load DICOM files from a folder .

    Args:
        folder_path (str): [Path to the folder containing DICOM files]
        verbose (bool, optional): [Whether to print extra information]. Defaults to False.

    Returns:
        list: [A list of DICOM objects]
    """
    # Load all DICOM files
    dcm_files = sorted(glob.glob(folder_path + "/*.dcm"))

    # Load all DICOM files
    dcm_data = [dcmread(dcm_file) for dcm_file in dcm_files]
    if verbose:
        print(f"Loaded {len(dcm_data)} DICOM files from {folder_path}")

    return dcm_data 
    

def _get_contour_idx(contour_path: str) -> int:
    """Returns the contour index from the contour file name.

    Args:
        contour_path (str): [Path to the contour file]

    Returns:
        int: [Contour index]
    """
    # Get filename of filepath
    filename = os.path.basename(contour_path)
    # Use regular expression to get third element of filename
    matches = re.findall(r'\d+', filename)
    return int(matches[1])
    

def load_contours(folder_path: str, size: (int, int), verbose: bool = False) -> list:
    """Load contour files from a folder.

    Args:
        folder_path (str): [Path to the folder containing contour files]
        size ((int, int)): [Size of the image mask]
        verbose (bool, optional): [Whether to print extra information]. Defaults to False.

    Returns:
        list: [A list of contour masks]
    """
    # Load all contour files
    icontour_files = sorted(glob.glob(folder_path + '/*icontour*.txt'))
    ocontour_files = sorted(glob.glob(folder_path + '/*ocontour*.txt'))  
    icontour_dict = {}
    ocontour_dict = {}
    
    # Save inner contour masks in a dictionary
    for ic in icontour_files:
        with open(ic, 'r') as fp:
            points = np.array([[float(n) for n in l.strip().split()] for l in fp if l.strip()])
            mask = np.zeros(size, dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            icontour_dict[_get_contour_idx(ic) - 1] = mask
    # Save outer contour files into dictionary
    for oc in ocontour_files:
        with open(oc, 'r') as fp:
            points = np.array([[float(n) for n in l.strip().split()] for l in fp if l.strip()])
            mask = np.zeros(size, dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            ocontour_dict[_get_contour_idx(oc) - 1] = mask
            
    if verbose:
        print(f"Loaded {len(icontour_files) + len(ocontour_files)} contour files from {folder_path}")
    return icontour_dict, ocontour_dict