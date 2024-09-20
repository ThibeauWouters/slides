$(function() {
    $('select[name="yearSelection"').change(function(){
        var dataUrl = $(this).data('url');
        dataUrl += '&year=' + $(this).val();
        window.location = dataUrl;
    });

    $('input.locatie-cb').change(function(){
        var count = ($('.locatie-cb:checked').size());
        if (count == 0) {
            $(this).prop('checked', true);
            return;
        } else {
            $('.' + $(this).attr('id')).toggle();
        }
    });

    if (pntGetCookie('acceptCookies') != 'yes') {
        $('#cookie-bar').show();
    }

    $('#cookieChoice, label[for="cookieChoice"]').click(function(){
        $('.cookieButton').removeAttr('disabled');
    });

    $('.glyphicon-search').click(function(){$('#searchForm').submit()});
});

jQuery.cookie = function(name, value, options) {
    if (typeof value != 'undefined') { // name and value given, set cookie
        options = options || {};
        if (value === null) {
            value = '';
            options.expires = -1;
        }
        var expires = '';
        if (options.expires && (typeof options.expires == 'number' || options.expires.toUTCString)) {
            var date;
            if (typeof options.expires == 'number') {
                date = new Date();
                date.setTime(date.getTime() + (options.expires * 24 * 60 * 60 * 1000));
            } else {
                date = options.expires;
            }
            expires = '; expires=' + date.toUTCString(); // use expires attribute, max-age is not supported by IE
        }
        // CAUTION: Needed to parenthesize options.path and options.domain
        // in the following expressions, otherwise they evaluate to undefined
        // in the packed version for some reason...
        var path = options.path ? '; path=' + (options.path) : '';
        var domain = options.domain ? '; domain=' + (options.domain) : '';
        var secure = options.secure ? '; secure' : '';
        document.cookie = [name, '=', encodeURIComponent(value), expires, path, domain, secure].join('');
    } else { // only name given, get cookie
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
};

function acceptCookies() {
    pntSetCookie('acceptCookies','yes','/', 365);
    removeCookieBar();
}
function removeCookieBar() {
    $('#cookie-bar').remove();
}

function pntSetCookie(name, value) {
    var pPath = pntIsDefined(arguments[2]) ? arguments[2] : null;
    var pExpire = pntIsDefined(arguments[3]) ? arguments[3] : null;

    if ((pPath != null) && (pExpire != null)) {
        $.cookie(name, value, { path: pPath, expires: pExpire });
        return;
    }

    if (pPath != null) {
        $.cookie(name, value, { path: pPath });
        return;
    }

    if (pExpire != null) {
        $.cookie(name, value, { expires: pExpire });
        return;
    }
}
function pntGetCookie(name) {
    return $.cookie(name);
}
function pntIsDefined() {
    if (!arguments[0] || typeof arguments[0] == "undefined") return false;
    return true;
}