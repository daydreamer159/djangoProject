from django.db import models

# Create your models here.
class ImgInfo(models.Model):
    id=models.AutoField(primary_key=True)
    img=models.ImageField()
    imginfo=models.CharField(max_length=100)
