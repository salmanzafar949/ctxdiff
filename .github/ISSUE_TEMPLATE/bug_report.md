---
name: Bug report
about: Something ctxdiff captured wrong, crashed on, or didn't record
title: "[bug] "
labels: bug
assignees: ''
---

**What happened**
A clear description of the bug.

**What you expected**
What you thought ctxdiff would do instead.

**Minimal reproduction**
The smallest snippet that shows it — ideally the `wrap()` call and the client you used:

```python
from ctxdiff import trace
tracer = trace.init("repro")
client = tracer.wrap(...)
# ...
```

If you can, attach or describe the resulting `.ctrace` (it's a SQLite file — `sqlite3 run.ctrace ".schema"` and a couple of rows help a lot).

**Environment**
- ctxdiff version:
- Python version:
- Provider / SDK + version (openai, anthropic, langchain, …):
- OS:

**Anything else**
Logs, stack traces, or context. Remember `ctxdiff` is fail-open — if capture silently produced nothing, note whether you saw a `ctxdiff:` warning in your logs.
