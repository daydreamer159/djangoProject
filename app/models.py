
# Create your models here.
from django.db import models

# Create your models here.

class Person(models.Model):
    # 姓名 str类型，verbose_name 是后台管理界面中显示的内容
    name = models.CharField(max_length=20,verbose_name='姓名')
    # 年龄 int型
    age=models.IntegerField(verbose_name='年龄')
    # 成绩 float类型
    score = models.FloatField(verbose_name='成绩')

