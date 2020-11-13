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

const SpotifyWebApi = require("spotify-web-api-node")
var scopes = ['user-read-private', 'user-read-email'],
    redirectUri = 'http://localhost:8080/logged',
    clientId = process.env.CLIENT_ID,
    clientSecret = process.env.CLIENT_SECRET,
    state = 'some-state';

var spotifyApi = new SpotifyWebApi({
    redirectUri: redirectUri,
    clientId: clientId,
    clientSecret: clientSecret
});

var authorizeURL = spotifyApi.createAuthorizeURL(scopes, state);

app.get('/test', (req, res) => {
    console.log("called")
    res.send('Welcome to the backend!')
})

//app.get('/', (req, res) => res.send('Home Route'));
app.get("/", async (req, res) => [
    //console.log(res),
    res.redirect(authorizeURL)])

function refreshAccessToken() {
    spotifyApi.refreshAccessToken().then(
        function (data) {
            console.log('The access token has been refreshed!');

            // Save the access token so that it's used in future calls
            spotifyApi.setAccessToken(data.body['access_token']);
        },
        function (err) {
            console.log('Could not refresh access token', err);
        }
    );
}

//app.get("/logged", async (req, res) => res.send("Successful log-in"))
app.get("/logged", async (req, res) => {
    let code = req.query.code
    console.log(code)
    spotifyApi.authorizationCodeGrant(code).then(
        function (data) {
            if (data.body['expires_in'] < "1500") {
                refreshAccessToken()
            }
            console.log('token expires in: ' + data.body['expires_in'])
            console.log('access token is: ' + data.body['access_token'])
            console.log('refresh token is ' + data.body['refresh_token'])
            spotifyApi.setAccessToken(data.body['access_token'])
            spotifyApi.setRefreshToken(data.body['refresh_token'])
        },
        function (err) {
            console.log('Something went wrong!', err)
        }
    )
    res.send("successful")
})



module.exports = spotifyConnect;