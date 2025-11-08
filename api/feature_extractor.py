"""
Feature Extractor for Phishing Detection
Extracts URL and content-based features from websites
"""

from urllib.parse import urlparse, parse_qs
import re
from typing import Dict, List
import numpy as np


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""

    def __init__(self):
        self.feature_names = [
            'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
            'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
            'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
            'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
            'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
            'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
            'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
            'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
            'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
            'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
            'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
            'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
            'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
            'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
            'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
        ]

        # Sensitive keywords often used in phishing
        self.sensitive_words = [
            'secure', 'account', 'update', 'confirm', 'login', 'signin',
            'banking', 'verify', 'suspended', 'locked', 'unusual', 'click'
        ]

        # Brand names often spoofed
        self.brand_names = [
            'paypal', 'amazon', 'ebay', 'google', 'facebook', 'microsoft',
            'apple', 'netflix', 'instagram', 'twitter', 'linkedin', 'yahoo'
        ]

    def extract_features(self, url: str, html_content: str = None) -> Dict[str, float]:
        """
        Extract all features from a URL

        Args:
            url: The URL to analyze
            html_content: Optional HTML content for content-based features

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        try:
            parsed = urlparse(url)

            # URL-based features
            features['NumDots'] = url.count('.')
            features['SubdomainLevel'] = len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0
            features['PathLevel'] = len([p for p in parsed.path.split('/') if p])
            features['UrlLength'] = len(url)
            features['NumDash'] = url.count('-')
            features['NumDashInHostname'] = parsed.netloc.count('-')
            features['AtSymbol'] = 1 if '@' in url else 0
            features['TildeSymbol'] = 1 if '~' in url else 0
            features['NumUnderscore'] = url.count('_')
            features['NumPercent'] = url.count('%')
            features['NumQueryComponents'] = len(parse_qs(parsed.query))
            features['NumAmpersand'] = url.count('&')
            features['NumHash'] = url.count('#')
            features['NumNumericChars'] = sum(c.isdigit() for c in url)
            features['NoHttps'] = 0 if parsed.scheme == 'https' else 1

            # Check for random strings (high entropy in subdomain/path)
            features['RandomString'] = self._check_random_string(url)

            # IP address instead of domain
            features['IpAddress'] = 1 if self._is_ip_address(parsed.netloc) else 0

            # Domain characteristics
            domain = self._get_domain(parsed.netloc)
            features['DomainInSubdomains'] = 1 if domain in parsed.netloc.replace(domain, '', 1) else 0
            features['DomainInPaths'] = 1 if domain in parsed.path else 0
            features['HttpsInHostname'] = 1 if 'https' in parsed.netloc else 0
            features['HostnameLength'] = len(parsed.netloc)
            features['PathLength'] = len(parsed.path)
            features['QueryLength'] = len(parsed.query)
            features['DoubleSlashInPath'] = 1 if '//' in parsed.path else 0

            # Sensitive words
            url_lower = url.lower()
            features['NumSensitiveWords'] = sum(1 for word in self.sensitive_words if word in url_lower)
            features['EmbeddedBrandName'] = sum(1 for brand in self.brand_names if brand in url_lower)

            # HTML content-based features (if available)
            if html_content:
                html_features = self._extract_html_features(html_content, parsed.netloc)
                features.update(html_features)
            else:
                # Default values for HTML features
                html_defaults = {
                    'PctExtHyperlinks': 0.0,
                    'PctExtResourceUrls': 0.0,
                    'ExtFavicon': 0,
                    'InsecureForms': 0,
                    'RelativeFormAction': 0,
                    'ExtFormAction': 0,
                    'AbnormalFormAction': 0,
                    'PctNullSelfRedirectHyperlinks': 0.0,
                    'FrequentDomainNameMismatch': 0,
                    'FakeLinkInStatusBar': 0,
                    'RightClickDisabled': 0,
                    'PopUpWindow': 0,
                    'SubmitInfoToEmail': 0,
                    'IframeOrFrame': 0,
                    'MissingTitle': 0,
                    'ImagesOnlyInForm': 0
                }
                features.update(html_defaults)

            # Risk threshold features (RT)
            features['SubdomainLevelRT'] = self._calculate_rt(features['SubdomainLevel'], [0, 1, 2])
            features['UrlLengthRT'] = self._calculate_rt(features['UrlLength'], [54, 75])
            features['PctExtResourceUrlsRT'] = self._calculate_rt(features['PctExtResourceUrls'], [0.22, 0.61])
            features['AbnormalExtFormActionR'] = 1 if features['ExtFormAction'] == 1 and features['AbnormalFormAction'] == 1 else 0
            features['ExtMetaScriptLinkRT'] = -1  # Placeholder
            features['PctExtNullSelfRedirectHyperlinksRT'] = self._calculate_rt(
                features['PctNullSelfRedirectHyperlinks'], [0.0, 0.17]
            )

        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features on error
            features = {name: 0.0 for name in self.feature_names}

        return features

    def _check_random_string(self, url: str) -> int:
        """Check if URL contains random-looking strings"""
        # Simple heuristic: check for long sequences of consonants or mixed case
        patterns = [
            r'[bcdfghjklmnpqrstvwxyz]{8,}',  # Many consonants
            r'[A-Z][a-z][A-Z][a-z][A-Z]',     # Alternating case
        ]
        for pattern in patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return 1
        return 0

    def _is_ip_address(self, netloc: str) -> bool:
        """Check if netloc is an IP address"""
        # Remove port if present
        host = netloc.split(':')[0]
        # Simple IPv4 check
        parts = host.split('.')
        if len(parts) == 4:
            try:
                return all(0 <= int(part) <= 255 for part in parts)
            except ValueError:
                return False
        return False

    def _get_domain(self, netloc: str) -> str:
        """Extract main domain from netloc"""
        parts = netloc.split('.')
        if len(parts) >= 2:
            return parts[-2]
        return netloc

    def _extract_html_features(self, html: str, base_domain: str) -> Dict[str, float]:
        """Extract features from HTML content"""
        features = {}
        html_lower = html.lower()

        # Count links
        links = re.findall(r'href=["\']([^"\']+)["\']', html_lower)
        total_links = len(links)
        ext_links = sum(1 for link in links if base_domain not in link and link.startswith('http'))

        features['PctExtHyperlinks'] = ext_links / total_links if total_links > 0 else 0.0

        # External resources
        resources = re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html_lower)
        total_resources = len(resources)
        ext_resources = sum(1 for res in resources if base_domain not in res and res.startswith('http'))

        features['PctExtResourceUrls'] = ext_resources / total_resources if total_resources > 0 else 0.0

        # Forms
        features['InsecureForms'] = 1 if re.search(r'<form[^>]*>', html_lower) and 'https' not in html_lower else 0
        features['ExtFormAction'] = 1 if re.search(r'<form[^>]*action=["\']http', html_lower) and base_domain not in html_lower else 0
        features['RelativeFormAction'] = 1 if re.search(r'<form[^>]*action=["\'][^h]', html_lower) else 0
        features['AbnormalFormAction'] = 1 if re.search(r'<form[^>]*action=["\']#', html_lower) else 0
        features['SubmitInfoToEmail'] = 1 if 'mailto:' in html_lower else 0

        # JavaScript features
        features['RightClickDisabled'] = 1 if 'event.button==2' in html_lower or 'oncontextmenu' in html_lower else 0
        features['PopUpWindow'] = 1 if 'window.open' in html_lower or 'popup' in html_lower else 0

        # Content features
        features['IframeOrFrame'] = 1 if '<iframe' in html_lower or '<frame' in html_lower else 0
        features['MissingTitle'] = 1 if '<title></title>' in html_lower or '<title>' not in html_lower else 0
        features['ImagesOnlyInForm'] = 0  # Complex to determine

        # Null/self redirect
        null_redirects = html_lower.count('href="#"') + html_lower.count("href='#'")
        features['PctNullSelfRedirectHyperlinks'] = null_redirects / total_links if total_links > 0 else 0.0

        # Domain mismatch
        features['FrequentDomainNameMismatch'] = 1 if len(set(re.findall(r'https?://([^/]+)', html_lower))) > 3 else 0
        features['FakeLinkInStatusBar'] = 0  # Requires JavaScript analysis

        # Favicon
        features['ExtFavicon'] = 1 if re.search(r'<link[^>]*rel=["\'].*icon["\'][^>]*href=["\']http', html_lower) and base_domain not in html_lower else 0

        return features

    def _calculate_rt(self, value: float, thresholds: List[float]) -> int:
        """Calculate risk threshold feature"""
        if len(thresholds) == 2:
            if value < thresholds[0]:
                return -1
            elif value <= thresholds[1]:
                return 0
            else:
                return 1
        elif len(thresholds) == 3:
            if value <= thresholds[0]:
                return -1
            elif value <= thresholds[1]:
                return 0
            else:
                return 1
        return 0

    def get_feature_vector(self, url: str, html_content: str = None) -> np.ndarray:
        """
        Get feature vector as numpy array in correct order

        Args:
            url: The URL to analyze
            html_content: Optional HTML content

        Returns:
            numpy array of features
        """
        features = self.extract_features(url, html_content)
        return np.array([features.get(name, 0.0) for name in self.feature_names])
