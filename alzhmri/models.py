from django.db import models


class Prediction(models.Model):
    image = models.ImageField(upload_to="mri_images/")
    result = models.CharField(max_length=100)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.result} - {self.created_at}"