```
mamba create -n craftax-web python=3.10 pip wheel -y
mamba activate craftax-web
mamba env update -f env.yaml
```


Things that could be improved:
1. I've only set-up saving data to google cloud storage. This is easy to use but very cumbersome and confusing to set up. Is there a better default choice to suggest for future users?