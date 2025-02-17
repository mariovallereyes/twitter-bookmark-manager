# /core/universal_bookmark_extractor.py

import logging
from typing import Dict, Any
from flask import jsonify

logger = logging.getLogger(__name__)

class BookmarkExtractor:
    """Universal Twitter Bookmark Extractor."""
    
    def __init__(self):
        self.logger = logger
        self.script = None

    def get_extraction_script(self):
        """Returns the JavaScript code that will be injected to extract bookmarks"""
        
        script = """
        async function extractBookmarks() {
            try {
                console.log("Starting bookmark extraction...");
                
                // Check if we're on the login page
                if (window.location.href.includes('/login') || window.location.href.includes('/flow/login')) {
                    throw new Error("Not logged in. Please log in to Twitter first.");
                }
                
                // Get auth token from current session
                const cookieString = document.cookie;
                if (!cookieString) {
                    throw new Error("No cookies found. Please make sure you're logged in.");
                }
                
                const tokenCookie = cookieString.split('; ').find(row => row.startsWith('ct0='));
                if (!tokenCookie) {
                    throw new Error("Authentication token not found. Please make sure you're logged in.");
                }
                
                const token = tokenCookie.split('=')[1];
                
                let allBookmarks = [];
                let cursor = null;
                
                do {
                    // Prepare request parameters
                    const variables = {
                        count: 50,
                        cursor: cursor,
                        includePromotedContent: false
                    };
                    
                    const features = {
                        responsive_web_graphql_timeline_navigation_enabled: true
                    };
                    
                    console.log(`Fetching bookmarks page${cursor ? ' after ' + cursor : ''}...`);
                    
                    // Make request to Twitter's internal API
                    const response = await fetch(
                        'https://twitter.com/i/api/graphql/bookmark_timeline?' + 
                        `variables=${encodeURIComponent(JSON.stringify(variables))}&` +
                        `features=${encodeURIComponent(JSON.stringify(features))}`, 
                        {
                            headers: {
                                'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs=1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
                                'x-csrf-token': token,
                                'x-twitter-active-user': 'yes',
                                'x-twitter-auth-type': 'OAuth2Session'
                            }
                        }
                    );
                    
                    if (!response.ok) {
                        if (response.status === 429) {
                            throw new Error("Rate limit exceeded. Please try again later.");
                        }
                        if (response.status === 401) {
                            throw new Error("Unauthorized. Please log in to Twitter.");
                        }
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Process timeline instructions
                    const instructions = data.data.bookmark_timeline_v2.timeline.instructions;
                    const entries = instructions[0].entries;
                    
                    // Extract bookmark data
                    const pageBookmarks = entries
                        .filter(entry => entry.content.itemContent)
                        .map(entry => {
                            const tweet = entry.content.itemContent.tweet_results.result;
                            return {
                                tweet_id: tweet.rest_id,
                                tweet_url: `https://twitter.com/i/status/${tweet.rest_id}`,
                                full_text: tweet.legacy.full_text,
                                created_at: tweet.legacy.created_at,
                                user: {
                                    id: tweet.core.user_results.result.rest_id,
                                    name: tweet.core.user_results.result.legacy.name,
                                    screen_name: tweet.core.user_results.result.legacy.screen_name,
                                    profile_image_url: tweet.core.user_results.result.legacy.profile_image_url_https
                                },
                                media: tweet.legacy.extended_entities?.media || [],
                                bookmark_date: new Date().toISOString()
                            };
                        });
                    
                    allBookmarks = allBookmarks.concat(pageBookmarks);
                    console.log(`Processed ${pageBookmarks.length} bookmarks from current page`);
                    
                    // Get cursor for next page
                    const nextCursor = entries[entries.length - 1]?.content?.value;
                    cursor = nextCursor;
                    
                    // If no next cursor or we've reached the end, stop
                    if (!nextCursor || entries.length < 2) {
                        break;
                    }
                    
                } while (true);
                
                console.log(`Finished extracting ${allBookmarks.length} bookmarks`);
                return allBookmarks;
                
            } catch (error) {
                console.error("Error extracting bookmarks:", error);
                throw error;
            }
        }
        
        // Start extraction when called
        return extractBookmarks();
        """
        
        return script

    def get_injection_wrapper(self):
        """Returns a wrapper object containing the script and metadata"""
        return {
            'status': 'success',
            'script': self.get_extraction_script(),
            'message': 'Extraction script ready'
        }