# Security Scanning

This document describes the security scanning setup for the SVG-AI project.

## Overview

The project uses `pip-audit` for automated dependency vulnerability scanning to identify and address security issues in Python packages. Security scanning is integrated into both development workflow and CI/CD pipeline.

## Tools

### pip-audit
- **Purpose**: Scans Python dependencies for known security vulnerabilities
- **Source**: PyPI Security Advisory Database and OSV.dev
- **Frequency**: Daily automated scans + on-demand manual scans
- **Output**: JSON and Markdown reports with vulnerability details

## Security Scan Results

### Current Status
✅ **23 vulnerabilities fixed** (setuptools, transformers)
⚠️ **4 remaining vulnerabilities** in torch (awaiting upstream fixes)

### Fixed Vulnerabilities
- **setuptools**: 78.1.0 → 80.9.0 (1 vulnerability fixed)
- **transformers**: 4.36.0 → 4.56.2 (22 vulnerabilities fixed)
- **torch**: 2.1.0 → 2.2.2 (19 vulnerabilities partially fixed)

### Remaining Vulnerabilities
All remaining vulnerabilities are in `torch` and require versions not yet available:
- PYSEC-2025-41: Requires torch ≥ 2.6.0
- PYSEC-2024-259: Requires torch ≥ 2.5.0
- GHSA-3749-ghw9-m3mg: Requires torch ≥ 2.7.1rc1
- GHSA-887c-mr87-cxwp: Requires torch ≥ 2.8.0

**Risk Assessment**: Low impact - these are primarily DoS vulnerabilities in ML training contexts, not affecting SVG conversion functionality.

## Manual Security Scanning

### Development Environment
```bash
# Activate virtual environment
source venv39/bin/activate

# Run basic security scan
pip-audit

# Generate detailed reports
pip-audit --format=json --output=security-report.json
pip-audit --format=markdown --output=security-report.md

# Scan specific requirements file
pip-audit -r requirements/prod.txt

# Check for high-severity issues only
pip-audit --format=json | jq '.vulnerabilities[] | select(.severity == "high" or .severity == "critical")'
```

### CI/CD Integration
Security scanning is automated through GitHub Actions (`.github/workflows/security-scan.yml`):

**Triggers**:
- Push to main/master/develop branches
- Pull requests to main/master
- Daily scheduled scan at 2 AM UTC
- Manual workflow dispatch

**Actions**:
- Install development dependencies
- Run pip-audit security scan
- Generate JSON and Markdown reports
- Upload scan results as artifacts
- Fail build on high/critical severity vulnerabilities
- Comment scan results on pull requests
- Dependency review for PR changes

## Security Policies

### Vulnerability Response
1. **Critical/High Severity**: Fix within 24 hours
2. **Moderate Severity**: Fix within 1 week
3. **Low Severity**: Fix in next maintenance cycle
4. **Informational**: Review and document

### Update Strategy
1. **Security Updates**: Apply immediately when available
2. **Major Version Upgrades**: Test thoroughly before applying
3. **Compatibility**: Ensure all dependencies remain compatible
4. **Rollback Plan**: Keep previous working versions documented

### Dependencies with Ongoing Issues
- **torch**: Monitor for security releases, upgrade when available
- **AI/ML packages**: Extra scrutiny due to complex dependencies

## Monitoring and Alerts

### Automated Monitoring
- **Daily Scans**: Detect new vulnerabilities in existing dependencies
- **PR Checks**: Ensure new dependencies don't introduce vulnerabilities
- **Dependency Review**: GitHub's automated dependency change analysis

### Manual Reviews
- **Monthly**: Review all security scan results
- **Quarterly**: Full dependency audit and cleanup
- **Before Releases**: Complete security review

## Security Scan Artifacts

### Generated Reports
- `audit-results.json`: Machine-readable vulnerability data
- `audit-results.md`: Human-readable security report
- Available as GitHub Actions artifacts for 90 days

### Report Contents
- Package name and version
- Vulnerability ID (PYSEC, GHSA, CVE)
- Severity level
- Fixed versions available
- Vulnerability description and impact

## Integration with Development Workflow

### Pre-commit Hooks (Optional)
```bash
# Add to .pre-commit-config.yaml
- repo: https://github.com/pypa/pip-audit
  rev: v2.7.3
  hooks:
    - id: pip-audit
      args: [--format=json]
```

### IDE Integration
Many IDEs support pip-audit through plugins:
- **VS Code**: Python security linting extensions
- **PyCharm**: Security vulnerability inspection
- **Vim/Neovim**: ALE with pip-audit support

## Troubleshooting

### Common Issues

1. **False Positives**
   ```bash
   # Ignore specific vulnerabilities (use carefully)
   pip-audit --ignore-vuln PYSEC-2024-XXX
   ```

2. **Package Not Found**
   - Local packages (like svg-ai) are skipped automatically
   - Editable installs may not be audited

3. **Network Issues**
   ```bash
   # Use local vulnerability database
   pip-audit --local-db
   ```

4. **Version Conflicts**
   - Update conflicting packages together
   - Use pip-tools for better dependency resolution

### Performance Optimization
```bash
# Cache vulnerability database
pip-audit --cache-dir ~/.cache/pip-audit

# Parallel scanning for multiple requirements files
pip-audit -r requirements/base.txt &
pip-audit -r requirements/dev.txt &
pip-audit -r requirements/prod.txt &
wait
```

## Security Best Practices

### Development
- Run security scans before committing changes
- Keep dependencies up to date
- Use virtual environments for isolation
- Pin dependency versions for reproducibility

### Production
- Use production requirements with minimal dependencies
- Regularly update base Docker images
- Monitor for new vulnerability disclosures
- Implement security headers and best practices

### Reporting
- Document all security findings
- Track remediation efforts
- Share security updates with team
- Maintain security changelog

## Future Improvements

### Planned Enhancements
- **SBOM Generation**: Software Bill of Materials for compliance
- **SAST Integration**: Static Application Security Testing
- **Dependency Graph**: Visual dependency vulnerability mapping
- **Security Metrics**: Track vulnerability trends over time

### Tool Upgrades
- Monitor pip-audit releases for new features
- Consider additional security scanning tools
- Integrate with security platforms (Snyk, WhiteSource, etc.)
- Automated vulnerability remediation with dependabot