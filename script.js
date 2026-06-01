/* ── Auto-assign left/right alternation from document order ── */
document.querySelectorAll('.timeline-item').forEach((item, i) => {
  item.classList.add(i % 2 === 0 ? 'right' : 'left');
});

/* ── Entrance animations via IntersectionObserver ── */
const items = document.querySelectorAll('.timeline-item');
const years = document.querySelectorAll('.timeline-year');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.12 });

items.forEach(item => observer.observe(item));
years.forEach(year => observer.observe(year));
document.querySelectorAll('.timeline-month').forEach(m => observer.observe(m));

/* ── Fold / unfold misc cards ── */
document.querySelectorAll('.fold-toggle').forEach(toggle => {
  toggle.addEventListener('click', () => {
    const content = toggle.closest('.timeline-content');
    const nowFolded = content.classList.toggle('folded');
    toggle.textContent = nowFolded ? '∨' : '∧';
    toggle.setAttribute('aria-expanded', String(!nowFolded));
  });
});

/* ── Filter buttons ── */
const filterBtns = document.querySelectorAll('.filter-btn');
const timelineMonths = document.querySelectorAll('.timeline-month');
const timelineYears = document.querySelectorAll('.timeline-year');
const scopeToggle = document.querySelector('.scope-toggle');

// Map filter value to tag classes that match
const filterMap = {
  ai: ['tag-ai', 'tag-symolicai'],
  flutter: ['tag-crossplatform'],
  web: ['tag-web'],
  docs: ['tag-docs'],
  other: ['tag-scripts', 'tag-security', 'tag-systems', 'tag-sql','tag-kotlin'],
  // unique: ongoing
  ongoing: ['tag-ongoing', 'tag-current'],
};

let activeCategory = 'all';
let activeScope = 'all';
let startRecord = 0;

function setSideClasses(items, randomStartReassigned = 0) {
  let idx = randomStartReassigned; // Start from a random side to add some variety when filtering, but keep the original left/right as much as possible for "all"
  items.forEach(item => {
    item.classList.remove('left', 'right');
    item.classList.add( idx % 2 === 0 ? 'right' : 'left');
    idx++;
  });
}

function itemMatchesCategory(item, category) {
  if (category === 'all') return true;
  const tagClasses = filterMap[category] || [];
  return tagClasses.some(tc => item.querySelector('.tag.' + tc));
}

function itemMatchesScope(item, scope) {
  if (scope === 'all') return true;
  return item.querySelector('.tag.tag-' + scope);
}

function applyFilter(changedType) {
  let randomStartReassigned;

  if (activeCategory === 'all') {
    // Always right first for "all" category, regardless of scope
    randomStartReassigned = 0;
  } else if (changedType === 'category') {
    // New non-"all" category selected — randomize layout
    startRecord = Math.floor(Math.random() * 2);
    randomStartReassigned = startRecord;
  } else {
    // Only scope changed — preserve current layout
    randomStartReassigned = startRecord;
  }

  // Update button states
  filterBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === activeCategory);
  });

  // Update scope toggle visual
  if (scopeToggle) {
    scopeToggle.dataset.scope = activeScope;
    const labels = { all: '◎', personal: 'P', school: 'S' };
    scopeToggle.textContent = labels[activeScope];
    scopeToggle.title = 'Scope: ' + activeScope.charAt(0).toUpperCase() + activeScope.slice(1);
  }

  // Show/hide items
  const allItems = document.querySelectorAll('.timeline-item');
  allItems.forEach(item => {
    const categoryMatch = itemMatchesCategory(item, activeCategory);
    const scopeMatch = itemMatchesScope(item, activeScope);
    const visible = categoryMatch && scopeMatch;

    item.style.display = visible ? '' : 'none';
    if (visible) item.classList.add('visible');
  });

  const visibleItems = document.querySelectorAll('.timeline-item:not([style*="display: none"])');
  setSideClasses(visibleItems, randomStartReassigned);

  // Show/hide months (only if they have visible items after them)
  timelineMonths.forEach(month => {
    let hasVisible = false;
    let el = month.nextElementSibling;
    while (el && !el.classList.contains('timeline-month') && !el.classList.contains('timeline-year')) {
      if (el.classList.contains('timeline-item') && el.style.display !== 'none') {
        hasVisible = true;
        break;
      }
      el = el.nextElementSibling;
    }
    month.style.display = hasVisible ? '' : 'none';
    if (hasVisible) month.classList.add('visible');
  });

  // Show/hide years (only if they have visible items)
  timelineYears.forEach(year => {
    let hasVisible = false;
    let el = year.nextElementSibling;
    while (el && !el.classList.contains('timeline-year')) {
      if (el.classList.contains('timeline-item') && el.style.display !== 'none') {
        hasVisible = true;
        break;
      }
      el = el.nextElementSibling;
    }
    year.style.display = hasVisible ? '' : 'none';
    if (hasVisible) year.classList.add('visible');
  });

  return startRecord;
}

filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    activeCategory = btn.dataset.filter;
    startRecord = applyFilter('category');
  });
});

if (scopeToggle) {
  scopeToggle.addEventListener('click', () => {
    const cycle = ['all', 'personal', 'school'];
    const idx = cycle.indexOf(activeScope);
    activeScope = cycle[(idx + 1) % cycle.length];
    startRecord = applyFilter('scope');
  });
}

/* ── Scroll-to-top button ── */
const topBtn = document.getElementById('back-to-top');

window.addEventListener('scroll', () => {
  topBtn.classList.toggle('show', window.scrollY > 320);
}, { passive: true });

topBtn.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
