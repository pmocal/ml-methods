web: waitress-serve --listen "*:$PORT" --trusted-proxy '*' --cleanup-interval 3000 --channel-timeout 3000 app:app
worker: celery app:celery_app worker