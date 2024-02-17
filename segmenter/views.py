from django.shortcuts import render

# Create your views here.

# 1. home Page
def home(request):
    return render(request, "index.html")

# 2. sement section
def segmenter(request):
    return render(request, "segment.html")

# 3. Project section
def project(request):
    return render(request, "project.html")

def output(request):
    return render(request, "output.html")

def payment(request):
    return render(request, "payment.html")