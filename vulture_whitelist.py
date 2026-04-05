# Vulture whitelist — false positives go here.
#
# When vulture flags code that IS actually used (Qt overrides, framework
# callbacks, plugin entry points), add it below using attribute syntax:
#
#     _.paintEvent  # Qt override
#     _.closeEvent  # Qt override
#
# To generate candidates: run vulture with --make-whitelist, review each
# entry, and paste only genuine false positives here. Dead code that
# vulture correctly identified must be deleted, not whitelisted.
