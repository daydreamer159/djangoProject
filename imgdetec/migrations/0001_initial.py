# Generated by Django 4.1 on 2023-06-05 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImgInfo',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('img', models.ImageField(upload_to='')),
                ('imginfo', models.CharField(max_length=100)),
            ],
        ),
    ]