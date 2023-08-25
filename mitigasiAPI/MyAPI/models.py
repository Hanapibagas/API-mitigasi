from django.db import models

# Create your models here.
class Classification(models.Model):
	
	
	reportmsg=models.CharField(max_length=255)
	
	def __str__(self):
		return self.reportmsg