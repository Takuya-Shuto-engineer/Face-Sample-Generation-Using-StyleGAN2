from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from .utils import get_image
 
def index(request):
    group_name = "johnnys"
    n_samples = 3
    img = get_image(group_name, n_samples)
    variables = {
        "app" : "face-sample-demo",
        "img" : img
    }
    return render(request, 'index.html', variables)
