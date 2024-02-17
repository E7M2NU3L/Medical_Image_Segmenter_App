from django.urls import path
from .views import home, segmenter, project, output, payment

# url patterns
urlpatterns = [
    path("", home, name="home_page"),
    path("segment/", segmenter, name="segment_app"), 
    path("project/", project, name="project_report"),
    path("segment/output/", output, name="output_file"),
    path('/payment', payment, name="payment"),
]