web: waitress-serve --listen "*:$PORT" --trusted-proxy '*' --cleanup-interval 3000 --channel-timeout 3000 app:app
worker: celery --app app.celery_app purge
worker: celery --app app.celery_app worker --purge --concurrency 1 -E
