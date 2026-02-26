web: waitress-serve --listen "*:$PORT" --trusted-proxy '*' --cleanup-interval 3000 --channel-timeout 3000 app:app
worker: celery multi start w1 --app app.celery_app --loglevel INFO --pool solo