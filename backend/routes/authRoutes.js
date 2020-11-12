async function spotifyConnect(){
    const SpotifyWebApi = require("spotify-web-api-node")
    var scopes = ['user-read-private', 'user-read-email'],
        redirectUri = 'http://localhost:8080/logged'
    clientId = process.env.clientId,
        state = 'some-state';

    var spotifyApi = new SpotifyWebApi({
        redirectUri: redirectUri,
        clientId: clientId
    });

    var authorizeURL = spotifyApi.createAuthorizeURL(scopes, state);

    return new Promise(resolve => {
            resolve(authorizeURL);
    })
}

module.exports = spotifyConnect;