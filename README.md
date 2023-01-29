# Urdu OCR App

* Install dependencies
* Build "maskrcnn_benchmark"
* Run "sudo ufw allow 800" to open port 8000 by allowing it over the firewall
* Run app using "python manage.py runserver 10.17.10.9:8000"
* Access at [IP_ADDRRESS]:8000 on local machine

### Automation 
* Reference - [This tutorial](https://www.codewithharry.com/blogpost/django-deploy-nginx-gunicorn/)
* Serve our application using Gunicorn by firing the commands:
    * "gunicorn ocrapp.wsgi:application --bind 10.17.10.9:8000 --timeout 180"
* Now create a system socket file for gunicorn now:
    * "sudo vim /etc/systemd/system/gunicorn.socket"
* Paste following in it
```
    [Unit]
    Description=gunicorn socket

    [Socket]
    ListenStream=/run/gunicorn.sock

    [Install]
    WantedBy=sockets.target
```

* Next, we will create a service file for gunicorn
    * "sudo vim /etc/systemd/system/gunicorn.service"
* Paste following in it
```
[Unit]
Description=Gunicorn instance to serve OCR app
Requires=gunicorn.socket
After=network.target

[Service]
User=baadalvm
Group=www-data
WorkingDirectory=/home/baadalvm/gunicorn_app
Environment=PATH="/home/baadalvm/miniconda3/envs/cn/bin"
ExecStart=/home/baadalvm/miniconda3/envs/cn/bin/gunicorn \
        --access-logfile - \
        --workers 3 \
        --bind unix:/run/gunicorn.sock \
        ocrapp.wsgi:application

[Install]
WantedBy=multi-user.target
```

* Lets now start and enable the gunicorn socket
    * sudo systemctl start gunicorn.socket
    * sudo systemctl enable gunicorn.socket

* Now we have to configure Nginx as a reverse proxy
* Create a configuration file for Nginx using the following command
    * "sudo vim /etc/nginx/sites-available/ocrapp"
* Paste the following in it
```
server {
        listen 80;
        server_name http://10.17.10.9;
        location / {
        include proxy_params;
            proxy_pass http://unix:/run/gunicorn.sock;
        }   
}
```
* Activate the configuration using the following command:
    * "sudo ln -s /etc/nginx/sites-available/ocrapp /etc/nginx/sites-enabled/"
* You may need to delete "sites-enabled" before previous step (Do once if it doesn't work)
* Restart nginx and allow the changes to take place
    * sudo systemctl restart nginx
* App should be available at "http://10.17.10.9"