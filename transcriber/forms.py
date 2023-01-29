from django import forms
from django.core.exceptions import ValidationError


def validate_txt_file(value):
    value= str(value)
    if value!= "transcription.txt": 
        raise ValidationError("Only TXT File named \"transcription.txt\" (as downloaded) can be uploaded")
    else:
        return value

def validate_zip_file(value):
    value= str(value)
    if (not value.endswith(".zip")) or (value == "predictions.zip"): 
        raise ValidationError("Please upload a valid ZIP file, as suggested!")
    else:
        return value

def validate_correction_zip_file(value):
    value= str(value)
    if  (value != "predictions.zip"): 
        raise ValidationError("Please upload a valid file, as suggested!")
    else:
        return value

def validate_uploaded_file(value):
    value = str(value)
    if not (value.endswith(".zip") or value.endswith(".jpg") or value.endswith(".png") or value.endswith(".JPG") or value.endswith(".PNG")):
        raise ValidationError("Please upload a valid file (Only ZIP, JPG & PNG files are allowed!)")
    else:
        return value

class FileUploadForm(forms.Form):
    file = forms.FileField(validators=[validate_uploaded_file])

class ImageUploadForm(forms.Form):
    image = forms.ImageField();

class TxtUploadForm(forms.Form):
    file = forms.FileField(validators=[validate_txt_file])

#class ZipUploadForm(forms.Form):
    #file = forms.FileField(validators=[validate_zip_file])

class CorrectionZipUploadForm(forms.Form):
    file = forms.FileField(validators=[validate_correction_zip_file])