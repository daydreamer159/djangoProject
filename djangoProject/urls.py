"""
URL configuration for djangoProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include

import app.views
import imgdetec.views
from app.views import *
from imgdetec.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('imgtest/', app.views.imgtest,name='upload_image'),
    # path('imgtest/', app.views.upload_image,name='upload_image'),
    # '' 内容为空，可以理解为首页
    # path('test/', app.views.predict),
    path('imgdetec/',imgdetec.views.upload_image,name='imgdetec')

]
