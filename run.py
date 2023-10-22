import CONFIG as cfg


if cfg.METHOD == 1:
    from Methods.M1.generator import Generator

vis = Generator()
try:
    getattr(vis, cfg.MODE.lower())()
except KeyboardInterrupt:
    print()
    vis.save()

vis.connection.close()
