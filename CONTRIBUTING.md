# Contributing

## Delivery Discipline

Complete each project phase or other significant change with a commit and push
after its tests and static checks pass. Before staging, verify that the changes
contain no private hostnames, credentials, private keys, local absolute paths,
market data, generated datasets, or private artifacts.

Generated research outputs belong below ignored directories such as
`artifacts/`; secrets remain in Kubernetes or local environment variables.
