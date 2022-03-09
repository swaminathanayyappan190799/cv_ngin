from django import forms

class FileInputForm(forms.Form):
    inputfile = forms.FileField(
        label='Select a file',
        help_text='max. 42 megabytes'
    )