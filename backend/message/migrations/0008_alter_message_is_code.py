# Generated by Django 4.2.7 on 2024-02-22 07:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('message', '0007_message_is_code'),
    ]

    operations = [
        migrations.AlterField(
            model_name='message',
            name='is_code',
            field=models.BooleanField(default=False),
        ),
    ]
